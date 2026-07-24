####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated_warning. Retrieved 13/35 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_underlying_function. Retrieved 14/26 statements.


def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir issues a deprecation warning.'
    var_1 = 'repo'
    var_2 = 'project'
    assert var_2 == 1
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'post_gen_project'
    var_7 = False
    var_8 = []
    var_9 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_10 = 'always'
    var_11 = 0
    var_12 = var_3.category
    var_13 = bool(var_9)
    assert var_13 is True
    var_14 = '_run_hook_from_repo_dir'
    var_15 = 'cookiecutter.hooks.run_hook_from_repo_dir'

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir calls run_hook_from_repo_dir with correct args.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'post_gen_project'
    var_9 = True
    var_10 = []
    var_11 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_12 = 'always'
    var_13 = len(var_10)
    assert var_13 == 1
    var_14 = var_10[0]['repo_dir']
    var_15 = var_10[0]['hook_name']
    var_16 = bool(var_10[0]['hook_name'] == var_8)
    assert var_16 is True
    var_17 = var_10[0]['project_dir']
    var_18 = var_10[0]['context']
    var_19 = bool(var_10[0]['context'] == var_7)
    assert var_19 is True
    var_20 = var_10[0]['delete_project_on_failure']
    assert var_20 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generate_context_with_valid_json_file. Retrieved 7/14 statements.
# Partially parsed test_generate_context_with_invalid_json_file. Retrieved 4/10 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 9/15 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 9/15 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 9/15 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 7/13 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 11/17 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 11/19 statements.
# Partially parsed test_generate_context_with_invalid_choice. Retrieved 10/16 statements.
# Partially parsed test_generate_context_with_invalid_boolean_conversion. Retrieved 8/14 statements.
# Partially parsed test_generate_context_preserves_context_structure. Retrieved 13/19 statements.


def test_case_0():
    var_0 = 'Test generate_context loads a valid JSON file correctly.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'my_project'
    var_5 = 'John Doe'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'cookiecutter'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ContextDecodingException for invalid JSON.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json content'
    var_3 = module_0.generate_context(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'ContextDecodingException'

def test_case_0():
    var_0 = 'Test generate_context applies default_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'default_project'
    var_5 = '1.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'overridden_project'
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'Test generate_context applies extra_context overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'default_project'
    var_5 = '1.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '2.0'
    var_8 = {var_3: var_7}

def test_case_0():
    var_0 = 'Test generate_context handles choice variables correctly.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = 'GPL'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_2: var_4}

def test_case_0():
    var_0 = 'Test generate_context converts string to boolean for boolean variables.'
    var_1 = 'cookiecutter.json'
    var_2 = 'use_docker'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'false'
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'Test generate_context handles nested dictionary variables.'
    var_1 = 'cookiecutter.json'
    var_2 = 'options'
    var_3 = 'debug'
    var_4 = 'verbose'
    var_5 = False
    var_6 = True
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = {var_3: var_6}
    var_10 = {var_2: var_9}

def test_case_0():
    var_0 = 'Test generate_context handles multichoice variables correctly.'
    var_1 = 'cookiecutter.json'
    var_2 = 'features'
    var_3 = 'feature1'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = [var_4, var_5]
    var_9 = {var_2: var_8}
    var_10 = 'cookiecutter'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ValueError for invalid choice.'
    var_1 = 'cookiecutter.json'
    var_2 = 'license'
    var_3 = 'MIT'
    var_4 = 'Apache'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = 'BSD'
    var_8 = {var_2: var_7}
    var_9 = module_0.generate_context(var_0, extra_context=var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'BSD'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ValueError for invalid boolean conversion.'
    var_1 = 'cookiecutter.json'
    var_2 = 'use_docker'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'maybe'
    var_6 = {var_2: var_5}
    var_7 = module_0.generate_context(var_0, extra_context=var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'could not be converted to a boolean'

def test_case_0():
    var_0 = 'Test generate_context preserves the original context structure.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'year'
    var_4 = 'enabled'
    var_5 = 'tags'
    var_6 = 'test'
    var_7 = 2024
    var_8 = True
    var_9 = 'tag1'
    var_10 = 'tag2'
    var_11 = [var_9, var_10]
    var_12 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_11}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_generate_context_json_decoding_error. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'Test that ValueError is caught and ContextDecodingException is raised at line 20.'
    var_1 = '{invalid json content}'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'JSON decoding error'



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
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
    var_0 = 'name'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'name': 'new_value'})
    assert var_6 is True

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
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'valid choices are'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = 'option3'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_3}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'choice': ['option3', 'option1', 'option2']})
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
    var_10 = 'but the choices are'

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
    var_0 = 'items'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'new_string'
    var_7 = {var_0: var_6}
    var_8 = False
    var_9 = module_0.apply_overwrites_to_context(var_5, var_7, in_dictionary_variable=var_8)
    var_10 = bool(var_5 == {'items': ['a', 'b', 'c']})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'items'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'new_string'
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = True
    var_12 = module_0.apply_overwrites_to_context(var_7, var_10, in_dictionary_variable=var_11)
    var_13 = bool(var_7 == {'nested': {'items': 'new_string'}})
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 'choice'
    var_3 = 'flag'
    var_4 = 'opt1'
    var_5 = 'opt2'
    var_6 = [var_4, var_5]
    var_7 = True
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = 'false'
    var_12 = {var_2: var_5, var_3: var_11}
    var_13 = {var_1: var_12}
    var_14 = {var_0: var_13}
    var_15 = module_0.apply_overwrites_to_context(var_10, var_14)
    var_16 = var_10['level1']['level2']['choice']
    var_17 = bool(var_10['level1']['level2']['choice'] == ['opt2', 'opt1'])
    assert var_17 is True
    var_18 = var_10['level1']['level2']['flag']
    assert var_18 is False



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that line 57 evaluates to False when YesNoPrompt.process_response succeeds.'
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
    var_0 = "Test that line 57 evaluates to False when converting 'yes' to boolean."
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
    var_0 = "Test that line 57 evaluates to False when converting 'no' to boolean."
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
    var_0 = "Test that line 57 evaluates to False when converting '0' to boolean."
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = '0'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that line 57 evaluates to False when converting '1' to boolean."
    var_1 = 'flag'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = '1'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_generate_context_raises_context_decoding_exception_on_json_decode_error. Retrieved 4/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that ValueError during JSON decoding raises ContextDecodingException.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json content}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'JSON decoding error'
    var_6 = 'Decoding error details'



# Parsed testcases at query #7
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that boolean conversion succeeds and InvalidResponse is not raised.'
    var_1 = 'is_enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['is_enabled']
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that boolean conversion to False succeeds.'
    var_1 = 'is_enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['is_enabled']
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that InvalidResponse exception is caught at line 57.'
    var_1 = 'is_enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'invalid_value'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'could not be converted to a boolean'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 3/11 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/10 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 5/12 statements.
# Partially parsed test_render_and_create_dir_existing_dir_raises_exception. Retrieved 4/15 statements.
# Partially parsed test_render_and_create_dir_existing_dir_overwrite. Retrieved 4/14 statements.
# Partially parsed test_render_and_create_dir_nested_path. Retrieved 3/10 statements.
# Partially parsed test_render_and_create_dir_returns_tuple. Retrieved 5/17 statements.


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
    var_5 = bool(var_2)
    assert var_5 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'existing'
    var_2 = 'existing'
    var_3 = {}
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'existing'
    var_2 = {}
    var_3 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'nested/deep/dir'
    var_2 = {}

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'new_dir'
    var_2 = {}
    var_3 = 0
    var_4 = 1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated_warning. Retrieved 12/25 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_actual_function. Retrieved 12/21 statements.
# Partially parsed test_run_hook_from_repo_dir_with_different_hook_names. Retrieved 13/22 statements.
# Partially parsed test_run_hook_from_repo_dir_delete_on_failure_true. Retrieved 10/19 statements.
# Partially parsed test_run_hook_from_repo_dir_delete_on_failure_false. Retrieved 10/19 statements.


def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir issues a deprecation warning.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = 'post_gen_project'
    var_8 = False
    var_9 = 0
    var_10 = 'deprecated'
    var_11 = 'post_gen_project'
    var_12 = False

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir delegates to run_hook_from_repo_dir.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_10 = 'warnings.warn'
    var_11 = 'pre_prompt'

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with various hook names.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = 'warnings.warn'
    var_8 = 'pre_prompt'
    var_9 = 'post_gen_project'
    var_10 = 'pre_gen_project'
    var_11 = [var_8, var_9, var_10]
    var_12 = False

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = 'warnings.warn'
    var_8 = 'post_gen_project'
    var_9 = True

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=False.'
    var_1 = 'repo'
    var_2 = 'project'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_7 = 'warnings.warn'
    var_8 = 'post_gen_project'
    var_9 = False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_false. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = True
    var_3 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/7 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 5/9 statements.
# Partially parsed test_render_and_create_dir_existing_dir_without_overwrite. Retrieved 4/11 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 4/10 statements.
# Partially parsed test_render_and_create_dir_nested_path. Retrieved 3/7 statements.
# Partially parsed test_render_and_create_dir_none_dirname. Retrieved 3/8 statements.


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
    var_4 = '{{ project_name }}'

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
    var_2 = 'parent/child/nested'

import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = {}
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 3/11 statements.
# Partially parsed test_render_and_create_dir_creates_directory. Retrieved 3/10 statements.
# Partially parsed test_render_and_create_dir_with_template. Retrieved 5/12 statements.
# Partially parsed test_render_and_create_dir_existing_dir_raises_exception. Retrieved 5/16 statements.
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
    var_4 = '{{ project_name }}_dir'
    var_5 = bool(var_2)
    assert var_5 is True

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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_generate_context_json_decoding_error. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'Test that ValueError predicate at line 20 evaluates to True when JSON is invalid.'
    var_1 = '{invalid json content}'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'JSON decoding error'



# Parsed testcases at query #14
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that line 57 predicate evaluates to False when conversion succeeds.'
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
    var_0 = "Test that line 57 predicate evaluates to False when converting 'no' to False."
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
    var_0 = "Test that line 57 predicate evaluates to False when converting '0' to False."
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
    var_0 = "Test that line 57 predicate evaluates to False when converting '1' to True."
    var_1 = 'enabled'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = '1'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['enabled']
    assert var_7 is True



# Parsed testcases at query #15
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that line 57 predicate evaluates to False when conversion succeeds.'
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
    var_0 = "Test that line 57 predicate evaluates to False when converting 'yes' to boolean."
    var_1 = 'flag'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that line 57 predicate evaluates to False when converting 'true' to boolean."
    var_1 = 'enabled'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'true'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['enabled']
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that line 57 predicate evaluates to False when converting 'false' to boolean."
    var_1 = 'enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'false'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['enabled']
    assert var_7 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 4/10 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 7/13 statements.
# Partially parsed test_render_and_create_dir_existing_dir_overwrite_false. Retrieved 6/14 statements.
# Partially parsed test_render_and_create_dir_existing_dir_overwrite_true. Retrieved 5/12 statements.
# Partially parsed test_render_and_create_dir_with_nested_path. Retrieved 4/10 statements.
# Partially parsed test_render_and_create_dir_return_values_for_new_dir. Retrieved 4/10 statements.


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
    var_0 = 'Test that EmptyDirNameException is raised when dirname is None.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = None
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that render_and_create_dir creates a new directory.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that render_and_create_dir renders template in dirname.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = '{{project_name}}_dir'
    var_6 = 'my_project_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that OutputDirExistsException is raised when dir exists and overwrite is False.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'existing_dir'
    var_4 = True
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that render_and_create_dir returns existing dir when overwrite is True.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'existing_dir'
    var_4 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that render_and_create_dir creates nested directory structure.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'parent/child/grandchild'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that render_and_create_dir returns correct tuple for new directory.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = 'new_test_dir'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecated_function. Retrieved 15/20 statements.
# Partially parsed test_run_hook_from_repo_dir_passes_all_arguments. Retrieved 13/19 statements.
# Partially parsed test_run_hook_from_repo_dir_deprecation_warning_category. Retrieved 11/15 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir issues deprecation warning and calls the new function.'
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
    var_13 = "The '_run_hook_from_repo_dir' function is deprecated, use 'cookiecutter.hooks.run_hook_from_repo_dir' instead"
    var_14 = 2

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir passes all arguments correctly to the new function.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'cookiecutter.generate.warnings.warn'
    var_3 = '/template/dir'
    var_4 = 'pre_gen_project'
    var_5 = '/output/dir'
    var_6 = [var_5]
    var_7 = 'cookiecutter'
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = False
    var_13 = False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir issues DeprecationWarning with correct stacklevel.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'cookiecutter.generate.warnings.warn'
    var_3 = '/repo'
    var_4 = 'hook'
    var_5 = '/project'
    var_6 = 'cookiecutter'
    var_7 = {}
    var_8 = {var_6: var_7}
    var_9 = True
    var_10 = module_0._run_hook_from_repo_dir(var_3, var_4, var_5, var_8, var_9)



# Parsed testcases at query #18
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that the predicate at line 18 evaluates to False when file doesn't exist."
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = True
    var_4 = False
    assert var_4 is False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 17/42 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 13/35 statements.
# Partially parsed test_generate_files_skip_if_exists. Retrieved 17/42 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 16/41 statements.
# Partially parsed test_generate_files_binary_file. Retrieved 14/25 statements.


def test_case_0():
    var_0 = 'Test generate_files with basic template structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'README.md'
    var_4 = '# {{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    assert var_5 == '# my_project'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = [var_9]
    var_11 = [var_10]
    var_12 = 'output'
    var_13 = None
    var_14 = False
    var_15 = 'my_project'
    var_16 = 'my_project'
    var_17 = 'README.md'
    var_18 = bool(var_3)
    assert var_18 is True
    var_19 = var_4 / var_17

def test_case_0():
    var_0 = 'Test generate_files calls hooks when accept_hooks is True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'file.txt'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = (var_4, var_7)
    var_9 = [var_8]
    var_10 = [var_9]
    var_11 = 'output'
    var_12 = []
    var_13 = True
    var_14 = 'pre_gen_project'
    var_15 = bool('pre_gen_project' in var_12)
    assert var_15 is True
    var_16 = 'post_gen_project'
    var_17 = bool('post_gen_project' in var_12)
    assert var_17 is True

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists option.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.proj}}'
    var_3 = 'existing.txt'
    var_4 = 'original content'
    var_5 = 'cookiecutter'
    var_6 = 'proj'
    var_7 = 'project'
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = [var_9]
    var_11 = [var_10]
    var_12 = 'output'
    var_13 = True
    var_14 = 'existing content'
    var_15 = None
    var_16 = True
    var_17 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists option.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'config.txt'
    var_4 = 'config: {{cookiecutter.name}}'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'app'
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = [var_9]
    var_11 = [var_10]
    var_12 = 'output'
    var_13 = 'old config'
    var_14 = None
    var_15 = True
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files handles binary files correctly.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.proj}}'
    var_3 = 'image.png'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = b'fake binary content'
    var_6 = var_4 + var_5
    var_7 = 'cookiecutter'
    var_8 = 'proj'
    var_9 = 'myproj'
    var_10 = {var_8: var_9}
    var_11 = (var_7, var_10)
    var_12 = [var_11]
    var_13 = [var_12]
    var_14 = 'output'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_true. Retrieved 6/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 24 evaluates to True when directory exists.'
    var_1 = 'existing_project'
    var_2 = True
    var_3 = 'project_name'
    var_4 = {var_3: var_1}
    var_5 = module_0.Environment()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 4/8 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 6/10 statements.
# Partially parsed test_render_and_create_dir_existing_dir_without_overwrite. Retrieved 5/12 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 5/11 statements.
# Partially parsed test_render_and_create_dir_nested_path. Retrieved 4/8 statements.
# Partially parsed test_render_and_create_dir_with_complex_template. Retrieved 8/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that EmptyDirNameException is raised when dirname is empty.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = ''
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Error: directory name is empty'

import jinja2.environment as module_0

def test_case_0():
    var_0 = "Test that a new directory is created when it doesn't exist."
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that directory name is rendered from template.'
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '{{ project_name }}_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that OutputDirExistsException is raised when directory exists and overwrite is False.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'existing_dir'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that existing directory is handled when overwrite_if_exists is True.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'existing_dir'
    var_4 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that nested directory structure is created.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'parent/child/grandchild'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test rendering with complex template expressions.'
    var_1 = 'name'
    var_2 = 'version'
    var_3 = 'test'
    var_4 = '1'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{ name }}_v{{ version }}'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that new variables at first level are ignored.'
    var_1 = 'existing_var'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'new_var'
    var_5 = 'new_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_3, var_6)
    var_8 = bool(var_3 == {'existing_var': 'value'})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that new variables in nested dictionaries are added.'
    var_1 = 'nested'
    var_2 = 'existing_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'new_key'
    var_7 = 'new_value'
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_5, var_9)
    var_11 = bool(var_5 == {'nested': {'existing_key': 'value', 'new_key': 'new_value'}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test valid multichoice variable overwrite.'
    var_1 = 'choices'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = [var_3, var_4]
    var_8 = {var_1: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_6, var_8)
    var_10 = bool(var_6 == {'choices': ['b', 'c']})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test invalid multichoice variable overwrite raises ValueError.'
    var_1 = 'choices'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = 'd'
    var_8 = [var_3, var_7]
    var_9 = {var_1: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'multi-choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test valid single choice variable overwrite.'
    var_1 = 'choice'
    var_2 = 'default'
    var_3 = 'option1'
    var_4 = 'option2'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_1: var_3}
    var_8 = module_0.apply_overwrites_to_context(var_6, var_7)
    var_9 = bool(var_6 == {'choice': ['option1', 'default', 'option2']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test invalid single choice variable overwrite raises ValueError.'
    var_1 = 'choice'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = 'd'
    var_8 = {var_1: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_6, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that nested dict overwrites work in dictionary variables.'
    var_1 = 'config'
    var_2 = 'nested'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'new_value'
    var_9 = {var_3: var_8}
    var_10 = {var_2: var_9}
    var_11 = {var_1: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_7, var_11)
    var_13 = bool(var_7 == {'config': {'nested': {'key': 'new_value'}}})
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test boolean variable overwrite with yes response.'
    var_1 = 'enabled'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'enabled': True})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test boolean variable overwrite with no response.'
    var_1 = 'enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'enabled': False})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test boolean variable overwrite with 'true' response."
    var_1 = 'flag'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'true'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'flag': True})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test boolean variable overwrite with 'false' response."
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'false'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'flag': False})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test invalid boolean conversion raises ValueError.'
    var_1 = 'enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'invalid'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'could not be converted to a boolean'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test simple string variable overwrite.'
    var_1 = 'name'
    var_2 = 'old_name'
    var_3 = {var_1: var_2}
    var_4 = 'new_name'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'name': 'new_name'})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test simple integer variable overwrite.'
    var_1 = 'count'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = 10
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'count': 10})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that list overwrites work in dictionary variables.'
    var_1 = 'config'
    var_2 = 'items'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'c'
    var_9 = 'd'
    var_10 = [var_8, var_9]
    var_11 = {var_2: var_10}
    var_12 = {var_1: var_11}
    var_13 = True
    var_14 = module_0.apply_overwrites_to_context(var_7, var_12, in_dictionary_variable=var_13)
    var_15 = bool(var_7 == {'config': {'items': ['c', 'd']}})
    assert var_15 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test overwriting multiple variables at once.'
    var_1 = 'var1'
    var_2 = 'var2'
    var_3 = 'var3'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = 'value3'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'new1'
    var_9 = 'new3'
    var_10 = {var_1: var_8, var_3: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_7, var_10)
    var_12 = bool(var_7 == {'var1': 'new1', 'var2': 'value2', 'var3': 'new3'})
    assert var_12 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test partial overwrite of nested dictionary.'
    var_1 = 'config'
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'new_value1'
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_7, var_10)
    var_12 = bool(var_7 == {'config': {'key1': 'new_value1', 'key2': 'value2'}})
    assert var_12 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test boolean variable overwrite with '1' (true)."
    var_1 = 'enabled'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = '1'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'enabled': True})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test boolean variable overwrite with '0' (false)."
    var_1 = 'enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = '0'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'enabled': False})
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_apply_overwrites_to_context_boolean_with_various_true_values. Retrieved 3/6 statements.
# Partially parsed test_apply_overwrites_to_context_boolean_with_various_false_values. Retrieved 3/6 statements.


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
    var_0 = 'dict_var'
    var_1 = 'existing_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_key'
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = bool(var_4 == {'dict_var': {'existing_key': 'value', 'new_key': 'new_value'}})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'list_key'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 4
    var_9 = 5
    var_10 = 6
    var_11 = [var_8, var_9, var_10]
    var_12 = {var_1: var_11}
    var_13 = {var_0: var_12}
    var_14 = module_0.apply_overwrites_to_context(var_7, var_13)
    var_15 = bool(var_7 == {'dict_var': {'list_key': [4, 5, 6]}})
    assert var_15 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = {var_0: var_5}
    var_7 = [var_2, var_3]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_6, var_8)
    var_10 = bool(var_6 == {'choices': [2, 3]})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = [var_2, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'multi-choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = 'option3'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'choice': ['option2', 'option1', 'option3']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'option1'
    var_2 = 'option2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'option3'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': False})
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

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = bool(var_2 == {'flag': True})
    assert var_3 is True

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = bool(var_2 == {'flag': False})
    assert var_3 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'count'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = 10
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'count': 10})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_0.apply_overwrites_to_context(var_4, var_5)
    var_7 = bool(var_4 == {'var1': 'value1', 'var2': 'value2'})
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir_with_valid_dirname. Retrieved 5/9 statements.
# Partially parsed test_render_and_create_dir_empty_dirname_raises_exception. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_none_dirname_raises_exception. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_existing_dir_without_overwrite_raises_exception. Retrieved 6/13 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 6/11 statements.
# Partially parsed test_render_and_create_dir_nested_dirname. Retrieved 7/11 statements.
# Partially parsed test_render_and_create_dir_with_special_characters. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'Test render_and_create_dir creates directory with valid dirname.'
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ project_name }}'

def test_case_0():
    var_0 = 'Test render_and_create_dir raises EmptyDirNameException for empty dirname.'
    var_1 = {}
    var_2 = ''
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test render_and_create_dir raises EmptyDirNameException for None dirname.'
    var_1 = {}
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test render_and_create_dir raises OutputDirExistsException when dir exists and overwrite_if_exists is False.'
    var_1 = 'project_name'
    var_2 = 'existing_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ project_name }}'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'Test render_and_create_dir returns existing dir when overwrite_if_exists is True.'
    var_1 = 'project_name'
    var_2 = 'existing_project'
    var_3 = {var_1: var_2}
    var_4 = '{{ project_name }}'
    var_5 = True

def test_case_0():
    var_0 = 'Test render_and_create_dir creates nested directory structure.'
    var_1 = 'org'
    var_2 = 'project'
    var_3 = 'myorg'
    var_4 = 'myproject'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = '{{ org }}/{{ project }}'

def test_case_0():
    var_0 = 'Test render_and_create_dir with special characters in dirname.'
    var_1 = 'name'
    var_2 = 'test-project_v1'
    var_3 = {var_1: var_2}
    var_4 = '{{ name }}'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_true. Retrieved 4/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_both_defaults_and_extra. Retrieved 9/13 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_nested_dict. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_custom_filename. Retrieved 3/7 statements.
# Partially parsed test_generate_context_invalid_choice_raises_error. Retrieved 7/11 statements.
# Partially parsed test_generate_context_invalid_boolean_conversion. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'Test generate_context loads and returns context from JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context applies default_context overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "default_name", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'overwritten_name'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context applies extra_context overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "default_name", "version": "1.0"}'
    var_3 = 'version'
    var_4 = '2.0'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context applies both default and extra context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "default", "version": "1.0", "author": "unknown"}'
    var_3 = 'project_name'
    var_4 = 'from_default'
    var_5 = {var_3: var_4}
    var_6 = 'version'
    var_7 = '2.0'
    var_8 = {var_6: var_7}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ContextDecodingException for invalid JSON.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"invalid json}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'ContextDecodingException'
    var_6 = 'JSON decoding error'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test generate_context raises FileNotFoundError when file doesn't exist."
    var_1 = '/nonexistent/path/cookiecutter.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'Test generate_context handles choice variables with extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context handles multichoice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_3 = 'features'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test generate_context handles nested dictionary variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"key1": "value1", "key2": "value2"}}'
    var_3 = 'config'
    var_4 = 'key1'
    var_5 = 'overwritten'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test generate_context converts string to boolean for boolean variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true, "use_ci": false}'
    var_3 = 'use_docker'
    var_4 = 'use_ci'
    var_5 = 'false'
    var_6 = 'yes'
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = 'Test generate_context with custom context filename.'
    var_1 = 'custom.json'
    var_2 = '{"project_name": "test"}'
    var_3 = 'custom'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ValueError for invalid choice.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'GPL'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ValueError for invalid boolean conversion.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_feature": true}'
    var_3 = 'use_feature'
    var_4 = 'invalid_boolean'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'could not be converted to a boolean'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_render_and_create_dir_raises_on_empty_dirname. Retrieved 5/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that EmptyDirNameException is raised when dirname is empty string.'
    var_1 = {}
    var_2 = '.'
    var_3 = [var_2]
    var_4 = module_0.Environment()
    var_5 = ''
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #7
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 46 evaluates to False when conditions are not met.'
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



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_true. Retrieved 5/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 24 evaluates to True when directory exists.'
    var_1 = 'test_dir'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = True



# Parsed testcases at query #10
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
    var_7 = 'Error: directory name is empty'



# Parsed testcases at query #11
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that line 57 predicate evaluates to False when conversion succeeds.'
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
    var_0 = "Test that line 57 predicate evaluates to False for 'yes' conversion."
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
    var_0 = "Test that line 57 predicate evaluates to False for 'on' conversion."
    var_1 = 'feature'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'on'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['feature']
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that line 57 predicate evaluates to False for '0' conversion."
    var_1 = 'active'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = '0'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['active']
    assert var_7 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_render_and_create_dir_with_valid_dirname. Retrieved 6/13 statements.
# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_existing_dir_no_overwrite. Retrieved 7/15 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 6/15 statements.
# Partially parsed test_render_and_create_dir_with_nested_path. Retrieved 8/15 statements.
# Partially parsed test_render_and_create_dir_with_plain_dirname. Retrieved 5/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ project_name }}'
    var_5 = False

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
    var_0 = 'project_name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ project_name }}'
    var_5 = True
    var_6 = False
    var_7 = bool(False)
    assert var_7 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'my_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ project_name }}'
    var_5 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'org'
    var_1 = 'project'
    var_2 = 'myorg'
    var_3 = 'myproject'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Environment()
    var_6 = '{{ org }}/{{ project }}'
    var_7 = False

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = 'simple_project'
    var_3 = False
    var_4 = 'simple_project'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = bool(var_4 == {'nested': {'existing': 'value', 'new_var': 'new_value'}})
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
    var_6 = 'x'
    var_7 = 'y'
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
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'true'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'false'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid_bool'
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
    var_1 = 'items'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'x'
    var_8 = 'y'
    var_9 = [var_7, var_8]
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = False
    var_13 = module_0.apply_overwrites_to_context(var_6, var_11, in_dictionary_variable=var_12)
    var_14 = bool(var_6 == {'nested': {'items': ['x', 'y']}})
    assert var_14 is True

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
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 'level3'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value'
    var_8 = {var_2: var_7}
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_6, var_10)
    var_12 = bool(var_6 == {'level1': {'level2': {'level3': 'new_value'}}})
    assert var_12 is True

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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 20
    var_8 = {var_1: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_6, var_8)
    var_10 = bool(var_6 == {'a': 1, 'b': 20, 'c': 3})
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'templates/file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'templates/*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'src/file.py'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'templates/*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'static/style.css'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'static/*'
    var_4 = 'media/*'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.is_copy_only_path(var_0, var_7)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'media/image.png'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'static/*'
    var_4 = 'media/*'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.is_copy_only_path(var_0, var_7)
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'src/main.py'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'static/*'
    var_4 = 'media/*'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.is_copy_only_path(var_0, var_7)
    assert var_8 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = module_0.is_copy_only_path(var_0, var_3)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = {}
    var_2 = module_0.is_copy_only_path(var_0, var_1)
    assert var_2 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = []
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.is_copy_only_path(var_0, var_5)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'README.md'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'README.md'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'docs/index.html'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'docs/*.html'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 4/10 statements.
# Partially parsed test_render_and_create_dir_with_template_variable. Retrieved 6/12 statements.
# Partially parsed test_render_and_create_dir_existing_dir_raises_exception. Retrieved 6/14 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 6/14 statements.
# Partially parsed test_render_and_create_dir_with_nested_path. Retrieved 4/10 statements.
# Partially parsed test_render_and_create_dir_with_complex_template. Retrieved 9/15 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that EmptyDirNameException is raised when dirname is empty.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = ''
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that EmptyDirNameException is raised when dirname is None.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = None
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that a new directory is created and returns correct values.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that directory name is rendered with context variables.'
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '{{ project_name }}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that OutputDirExistsException is raised when directory exists and overwrite is False.'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'existing_dir'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that existing directory is handled correctly when overwrite_if_exists is True.'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'existing_dir'
    var_5 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that nested directories are created correctly.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'parent/child/grandchild'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that complex template expressions are rendered correctly.'
    var_1 = 'name'
    var_2 = 'version'
    var_3 = 'test'
    var_4 = '1'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Environment()
    var_7 = '{{ name }}_v{{ version }}'
    var_8 = 'test_v1'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_with_boolean_and_string_overwrite. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_dict_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_file_stem_extraction. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_multichoice_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_default_context_warns. Retrieved 9/19 statements.


def test_case_0():
    var_0 = 'Test generate_context loads a basic JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John Doe"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context applies default_context overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'project_name'
    var_4 = 'overwritten_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context applies extra_context overwrites.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "version": "1.0"}'
    var_3 = 'version'
    var_4 = '2.0'
    var_5 = {var_3: var_4}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context raises ContextDecodingException for invalid JSON.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"invalid": json}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'JSON decoding error'

def test_case_0():
    var_0 = 'Test generate_context converts string to boolean for boolean variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_feature": true}'
    var_3 = 'use_feature'
    var_4 = 'no'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context handles choice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}
    var_6 = 'MIT'
    var_7 = 'GPL'

def test_case_0():
    var_0 = 'Test generate_context handles nested dictionary variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"author": {"name": "John", "email": "john@example.com"}}'
    var_3 = 'author'
    var_4 = 'name'
    var_5 = 'Jane'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test generate_context extracts correct file stem as context key.'
    var_1 = 'custom_template.json'
    var_2 = '{"key": "value"}'
    var_3 = 'custom_template'

def test_case_0():
    var_0 = 'Test generate_context handles multichoice variables.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"features": ["feature1", "feature2", "feature3"]}'
    var_3 = 'features'
    var_4 = 'feature2'
    var_5 = 'feature3'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context warns on invalid default_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'always'
    var_4 = 'license'
    var_5 = 'InvalidChoice'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_2, var_6)
    var_8 = 0
    var_9 = 'Invalid default received'



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that new variables at first level are ignored.'
    var_1 = 'existing'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'new_variable'
    var_5 = 'new_value'
    var_6 = {var_4: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_3, var_6)
    var_8 = bool(var_3 == {'existing': 'value'})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that new variables in nested dict are added.'
    var_1 = 'nested'
    var_2 = 'existing'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'new_key'
    var_7 = 'new_value'
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_5, var_9)
    var_11 = bool(var_5 == {'nested': {'existing': 'value', 'new_key': 'new_value'}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that simple values are overwritten.'
    var_1 = 'variable'
    var_2 = 'old_value'
    var_3 = {var_1: var_2}
    var_4 = 'new_value'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'variable': 'new_value'})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that valid multichoice overwrite is applied.'
    var_1 = 'choices'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = [var_3, var_4]
    var_8 = {var_1: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_6, var_8)
    var_10 = bool(var_6 == {'choices': ['b', 'c']})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that invalid multichoice overwrite raises ValueError.'
    var_1 = 'choices'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = 'd'
    var_8 = [var_3, var_7]
    var_9 = {var_1: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'multi-choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that valid single choice overwrite is applied.'
    var_1 = 'choice'
    var_2 = 'default'
    var_3 = 'option1'
    var_4 = 'option2'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_1: var_3}
    var_8 = module_0.apply_overwrites_to_context(var_6, var_7)
    var_9 = bool(var_6 == {'choice': ['option1', 'default', 'option2']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that invalid single choice overwrite raises ValueError.'
    var_1 = 'choice'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = 'invalid'
    var_8 = {var_1: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_6, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that nested dictionaries are partially overwritten.'
    var_1 = 'config'
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'new_value1'
    var_9 = {var_2: var_8}
    var_10 = {var_1: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_7, var_10)
    var_12 = bool(var_7 == {'config': {'key1': 'new_value1', 'key2': 'value2'}})
    assert var_12 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean context with 'yes' string is converted to True."
    var_1 = 'is_enabled'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'is_enabled': True})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean context with 'no' string is converted to False."
    var_1 = 'is_enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'no'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'is_enabled': False})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean context with 'true' string is converted to True."
    var_1 = 'is_enabled'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'true'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'is_enabled': True})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that boolean context with 'false' string is converted to False."
    var_1 = 'is_enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'false'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'is_enabled': False})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that boolean context with invalid string raises ValueError.'
    var_1 = 'is_enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'invalid'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'could not be converted to a boolean'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that list in nested dict is overwritten when in_dictionary_variable is True.'
    var_1 = 'nested'
    var_2 = 'choices'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = 'x'
    var_10 = 'y'
    var_11 = [var_9, var_10]
    var_12 = {var_2: var_11}
    var_13 = {var_1: var_12}
    var_14 = module_0.apply_overwrites_to_context(var_8, var_13)
    var_15 = bool(var_8 == {'nested': {'choices': ['x', 'y']}})
    assert var_15 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that integer values are overwritten.'
    var_1 = 'count'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = 10
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = bool(var_3 == {'count': 10})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that empty overwrite context doesn't change context."
    var_1 = 'variable'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.apply_overwrites_to_context(var_3, var_4)
    var_6 = bool(var_3 == {'variable': 'value'})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that multiple overwrites are applied correctly.'
    var_1 = 'var1'
    var_2 = 'var2'
    var_3 = 'var3'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = 'value3'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'new1'
    var_9 = 'new3'
    var_10 = {var_1: var_8, var_3: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_7, var_10)
    var_12 = bool(var_7 == {'var1': 'new1', 'var2': 'value2', 'var3': 'new3'})
    assert var_12 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that nested empty dict can be overwritten.'
    var_1 = 'config'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'new_key'
    var_5 = 'new_value'
    var_6 = {var_4: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_3, var_7)
    var_9 = bool(var_3 == {'config': {'new_key': 'new_value'}})
    assert var_9 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_run_hook_from_repo_dir_deprecation_warning. Retrieved 13/19 statements.
# Partially parsed test_run_hook_from_repo_dir_calls_actual_function. Retrieved 13/17 statements.
# Partially parsed test_run_hook_from_repo_dir_with_delete_true. Retrieved 13/16 statements.
# Partially parsed test_run_hook_from_repo_dir_with_empty_context. Retrieved 9/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir raises a deprecation warning.'
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
    var_15 = 'run_hook_from_repo_dir'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that _run_hook_from_repo_dir calls the actual run_hook_from_repo_dir.'
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
    var_11 = False
    var_12 = module_0._run_hook_from_repo_dir(var_3, var_4, var_5, var_10, var_11)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with delete_project_on_failure=True.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'cookiecutter.generate.warnings.warn'
    var_3 = '/path/to/repo'
    var_4 = 'pre_prompt'
    var_5 = '/path/to/project'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'myproject'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True
    var_12 = module_0._run_hook_from_repo_dir(var_3, var_4, var_5, var_10, var_11)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test _run_hook_from_repo_dir with empty context.'
    var_1 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_2 = 'cookiecutter.generate.warnings.warn'
    var_3 = '/path/to/repo'
    var_4 = 'post_gen_project'
    var_5 = '/path/to/project'
    var_6 = {}
    var_7 = False
    var_8 = module_0._run_hook_from_repo_dir(var_3, var_4, var_5, var_6, var_7)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 4/9 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 4/12 statements.
# Partially parsed test_render_and_create_dir_with_template_variable. Retrieved 7/14 statements.
# Partially parsed test_render_and_create_dir_existing_dir_without_overwrite. Retrieved 6/14 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 6/13 statements.
# Partially parsed test_render_and_create_dir_nested_path. Retrieved 7/17 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that EmptyDirNameException is raised when dirname is empty.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = ''
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that EmptyDirNameException is raised when dirname is None.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = None
    var_4 = bool(False)
    assert var_4 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that a new directory is created and returns correct path and flag.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'test_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that directory name is rendered with template variables.'
    var_1 = 'project_name'
    var_2 = 'my_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = '{{project_name}}_dir'
    var_6 = 'my_project_dir'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that OutputDirExistsException is raised when directory exists and overwrite is False.'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'existing_dir'
    var_5 = False
    var_6 = bool(False)
    assert var_6 is True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that existing directory is handled when overwrite_if_exists is True.'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'existing_dir'
    var_5 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that nested directory paths are created correctly.'
    var_1 = {}
    var_2 = module_0.Environment()
    var_3 = 'parent/child/grandchild'
    var_4 = 'parent'
    var_5 = 'child'
    var_6 = 'grandchild'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_true. Retrieved 6/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 24 evaluates to True when directory exists.'
    var_1 = 'existing_dir'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = 'existing_dir'
    var_5 = True



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_render_and_create_dir_raises_empty_dir_name_exception. Retrieved 5/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that render_and_create_dir raises EmptyDirNameException when dirname is empty.'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = module_0.Environment()
    var_5 = ''
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Error: directory name is empty'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 12/34 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 13/31 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 13/30 statements.
# Partially parsed test_generate_files_with_subdirectories. Retrieved 13/36 statements.
# Partially parsed test_generate_files_binary_file. Retrieved 12/32 statements.
# Partially parsed test_generate_files_with_context_variables. Retrieved 16/33 statements.
# Partially parsed test_generate_files_returns_project_dir. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'Test generate_files with basic template structure.'
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
    var_13 = 'my_project'
    var_14 = '# my_project'

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = 'my_project'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = {var_8: var_6}
    var_10 = (var_7, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = True

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'existing.txt'
    var_4 = 'new content'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'my_project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = True

def test_case_0():
    var_0 = 'Test generate_files with nested directory structure.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'src'
    var_4 = 'main.py'
    var_5 = "print('{{cookiecutter.project_name}}')"
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'my_project'
    var_10 = {var_8: var_9}
    var_11 = (var_7, var_10)
    var_12 = [var_11]
    var_13 = [var_12]

def test_case_0():
    var_0 = 'Test generate_files with binary files.'
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

def test_case_0():
    var_0 = 'Test generate_files with multiple context variables.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_slug}}'
    var_3 = 'config.txt'
    var_4 = 'project: {{cookiecutter.project_name}}\nauthor: {{cookiecutter.author}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'project_slug'
    var_9 = 'author'
    var_10 = 'My Project'
    var_11 = 'my_project'
    var_12 = 'John Doe'
    var_13 = {var_7: var_10, var_8: var_11, var_9: var_12}
    var_14 = (var_6, var_13)
    var_15 = [var_14]
    var_16 = [var_15]
    var_17 = 'project: My Project'
    var_18 = 'author: John Doe'

def test_case_0():
    var_0 = 'Test that generate_files returns the project directory path.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'file.txt'
    var_4 = 'content'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_render_and_create_dir_raises_on_empty_dirname. Retrieved 4/11 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that render_and_create_dir raises EmptyDirNameException when dirname is empty.'
    var_1 = module_0.Environment()
    var_2 = {}
    var_3 = ''
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_render_and_create_dir_with_empty_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_with_none_dirname. Retrieved 3/8 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/7 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 5/9 statements.
# Partially parsed test_render_and_create_dir_existing_dir_no_overwrite. Retrieved 5/12 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 4/10 statements.
# Partially parsed test_render_and_create_dir_nested_directory. Retrieved 3/7 statements.
# Partially parsed test_render_and_create_dir_with_complex_template. Retrieved 7/11 statements.


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
    var_2 = 'new_dir'

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
    var_3 = True
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True

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
    var_6 = '{{ name }}-v{{ version }}'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generate_context_catches_json_decoding_error. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'Test that generate_context catches ValueError on invalid JSON and raises ContextDecodingException.'
    var_1 = '{ invalid json content }'
    var_2 = False
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'JSON decoding error'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 3/11 statements.
# Partially parsed test_render_and_create_dir_creates_new_directory. Retrieved 3/10 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 5/12 statements.
# Partially parsed test_render_and_create_dir_existing_directory_without_overwrite. Retrieved 5/16 statements.
# Partially parsed test_render_and_create_dir_existing_directory_with_overwrite. Retrieved 4/14 statements.
# Partially parsed test_render_and_create_dir_nested_directory. Retrieved 3/10 statements.


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
    var_2 = 'new_dir'

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
    var_1 = {}
    var_2 = 'existing_dir'
    var_3 = 'existing_dir'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True

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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_generate_context_predicate_line_38_evaluates_to_false. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 38 (if default_context:) evaluates to False.'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = bool(var_2)
    assert var_5 is True
    var_6 = 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_context_json_decoding_error. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'Test that ValueError is caught at line 20 and converted to ContextDecodingException.'
    var_1 = '{invalid json content}'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'JSON decoding error'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_render_and_create_dir_predicate_line_24_true. Retrieved 6/13 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 24 evaluates to True when directory exists.'
    var_1 = 'existing_project'
    var_2 = True
    var_3 = 'project_name'
    var_4 = {var_3: var_1}
    var_5 = module_0.Environment()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_context_predicate_line_18_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = "Test that the predicate at line 18 (with open(...)) evaluates to False when file doesn't exist."
    var_1 = 'non_existent_cookiecutter.json'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_context_raises_context_decoding_exception_on_invalid_json. Retrieved 4/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that ValueError from json.load is caught and ContextDecodingException is raised.'
    var_1 = 'cookiecutter.json'
    var_2 = '{invalid json content'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'JSON decoding error'
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_generate_file_renders_text_file. Retrieved 9/27 statements.
# Partially parsed test_generate_file_copies_binary_file. Retrieved 7/25 statements.
# Partially parsed test_generate_file_renders_filename. Retrieved 10/27 statements.
# Partially parsed test_generate_file_skips_if_exists. Retrieved 9/26 statements.
# Partially parsed test_generate_file_returns_if_filename_empty. Retrieved 7/23 statements.
# Partially parsed test_generate_file_preserves_newlines. Retrieved 7/24 statements.
# Partially parsed test_generate_file_uses_configured_newlines. Retrieved 9/25 statements.
# Partially parsed test_generate_file_preserves_permissions. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'Hello {{ name }}!'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = 'World'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.bin'
    var_3 = b'\x89PNG\r\n\x1a\n'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = '{{ filename }}.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter'
    var_5 = 'filename'
    var_6 = 'output'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'output.txt'

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'original'
    var_4 = 'existing content'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}
    var_8 = True

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'content'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'line1\nline2\n'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'line1'
    var_8 = 'line2'

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'line1\nline2\n'
    var_4 = 'cookiecutter'
    var_5 = '_new_lines'
    var_6 = '\r\n'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}

def test_case_0():
    var_0 = 'templates'
    var_1 = 'project'
    var_2 = 'test.txt'
    var_3 = 'content'
    var_4 = 493
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}



# Parsed testcases at query #22
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that line 57 predicate evaluates to False when YesNoPrompt.process_response succeeds.'
    var_1 = 'enabled'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'yes'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['enabled']
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = "Test that line 57 predicate evaluates to False when YesNoPrompt.process_response succeeds with 'no'."
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
    var_0 = "Test that line 57 predicate evaluates to False when YesNoPrompt.process_response succeeds with 'true'."
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
    var_0 = "Test that line 57 predicate evaluates to False when YesNoPrompt.process_response succeeds with 'false'."
    var_1 = 'flag'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'false'
    var_5 = {var_1: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_3, var_5)
    var_7 = var_3['flag']
    assert var_7 is False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generate_files_with_default_parameters. Retrieved 16/27 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 17/29 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 17/27 statements.
# Partially parsed test_generate_files_with_none_context. Retrieved 9/19 statements.
# Partially parsed test_generate_files_with_accept_hooks_false. Retrieved 16/28 statements.
# Partially parsed test_generate_files_with_keep_project_on_failure_true. Retrieved 17/27 statements.


def test_case_0():
    var_0 = 'Test generate_files with default parameters.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = False

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'my_project'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = '_jinja2_env_vars'
    var_8 = {}
    var_9 = {var_6: var_4, var_7: var_8}
    var_10 = {var_5: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = True
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = True
    var_16 = False

def test_case_0():
    var_0 = 'Test generate_files with None context.'
    var_1 = 'repo'
    var_2 = 'test_template'
    var_3 = 'output'
    var_4 = 'cookiecutter.generate.find_template'
    var_5 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_6 = None
    var_7 = lambda *args, **kwargs: var_6
    var_8 = False

def test_case_0():
    var_0 = 'Test generate_files with accept_hooks=False.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'test_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = []
    var_12 = 'cookiecutter.generate.find_template'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = False
    var_15 = len(var_11)
    assert var_15 == 0

def test_case_0():
    var_0 = 'Test generate_files with keep_project_on_failure=True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = None
    var_14 = lambda *args, **kwargs: var_13
    var_15 = True
    var_16 = False



# Parsed testcases at query #24
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = bool(var_2 == {'existing': 'value'})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.apply_overwrites_to_context(var_2, var_5, in_dictionary_variable=var_6)
    var_8 = bool(var_2 == {'existing': {}, 'new_var': 'new_value'})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'key': 'new_value'})
    assert var_6 is True

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
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'provided for multi-choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'choice': ['b', 'a', 'c']})
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
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new_value'
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = bool(var_4 == {'outer': {'inner': 'new_value'}})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner1'
    var_2 = 'inner2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value1'
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'outer': {'inner1': 'new_value1', 'inner2': 'value2'}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': False})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'true'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'false'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'flag': False})
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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'flag'
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
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_3, var_2]
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'outer': {'inner': ['b', 'a']}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.apply_overwrites_to_context(var_0, var_3)
    var_5 = bool(var_0 == {})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.apply_overwrites_to_context(var_2, var_3)
    var_5 = bool(var_2 == {'key': 'value'})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'var3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_3, var_1: var_4, var_2: var_7}
    var_9 = 'new_value1'
    var_10 = {var_0: var_9, var_2: var_6}
    var_11 = module_0.apply_overwrites_to_context(var_8, var_10)
    var_12 = bool(var_8 == {'var1': 'new_value1', 'var2': 'value2', 'var3': ['b', 'a']})
    assert var_12 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generate_context_with_default_context. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 38 evaluates to True when default_context is provided.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'My Project'
    var_5 = 'my_project'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'Default Project'
    var_8 = {var_2: var_7}
    var_9 = 'cookiecutter'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generate_files_with_valid_context. Retrieved 20/33 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 14/26 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 20/32 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 21/33 statements.
# Partially parsed test_generate_files_skip_if_file_exists. Retrieved 21/33 statements.
# Partially parsed test_generate_files_keep_project_on_failure. Retrieved 21/33 statements.
# Partially parsed test_generate_files_with_files_to_process. Retrieved 23/37 statements.


def test_case_0():
    var_0 = 'Test generate_files with valid context and template directory.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = 'os.walk'
    var_14 = '.'
    var_15 = []
    var_16 = []
    var_17 = (var_14, var_15, var_16)
    var_18 = [var_17]
    var_19 = False

def test_case_0():
    var_0 = 'Test generate_files with empty context.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter.generate.find_template'
    var_5 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_6 = 'os.walk'
    var_7 = '.'
    var_8 = []
    var_9 = []
    var_10 = (var_7, var_8, var_9)
    var_11 = [var_10]
    var_12 = None
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files with hooks enabled.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = 'os.walk'
    var_14 = '.'
    var_15 = []
    var_16 = []
    var_17 = (var_14, var_15, var_16)
    var_18 = [var_17]
    var_19 = True

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = 'os.walk'
    var_14 = '.'
    var_15 = []
    var_16 = []
    var_17 = (var_14, var_15, var_16)
    var_18 = [var_17]
    var_19 = True
    var_20 = False

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = 'os.walk'
    var_14 = '.'
    var_15 = []
    var_16 = []
    var_17 = (var_14, var_15, var_16)
    var_18 = [var_17]
    var_19 = True
    var_20 = False

def test_case_0():
    var_0 = 'Test generate_files with keep_project_on_failure flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = 'os.walk'
    var_14 = '.'
    var_15 = []
    var_16 = []
    var_17 = (var_14, var_15, var_16)
    var_18 = [var_17]
    var_19 = True
    var_20 = False

def test_case_0():
    var_0 = 'Test generate_files with files in the template directory.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'output'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = '_jinja2_env_vars'
    var_7 = 'my_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'cookiecutter.generate.find_template'
    var_12 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_13 = 'cookiecutter.generate.generate_file'
    var_14 = 'os.walk'
    var_15 = '.'
    var_16 = []
    var_17 = 'test.txt'
    var_18 = [var_17]
    var_19 = (var_15, var_16, var_18)
    var_20 = [var_19]
    var_21 = 'cookiecutter.generate.is_copy_only_path'
    var_22 = False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 14/31 statements.
# Partially parsed test_generate_file_text_file. Retrieved 13/30 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 12/30 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 15/31 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/27 statements.
# Partially parsed test_generate_file_detected_newline. Retrieved 10/25 statements.
# Partially parsed test_generate_file_template_syntax_error. Retrieved 10/24 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'binary_file.bin'
    var_2 = b'\x89PNG\r\n\x1a\n'
    var_3 = 'templates'
    var_4 = 'os.chdir'
    var_5 = None
    var_6 = 'is_binary'
    var_7 = True
    var_8 = 'shutil.copyfile'
    var_9 = 'shutil.copymode'
    var_10 = module_0.Environment()
    var_11 = 'cookiecutter'
    var_12 = {}
    var_13 = {var_11: var_12}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'template_{{ cookiecutter.name }}.txt'
    var_2 = 'template_test.txt'
    var_3 = 'templates'
    var_4 = 'Hello {{ cookiecutter.name }}!'
    assert var_4 == 'Hello test!'
    var_5 = 'is_binary'
    var_6 = False
    var_7 = module_0.Environment()
    var_8 = 'cookiecutter'
    var_9 = 'name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = {var_8: var_11}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'existing_file.txt'
    var_2 = 'templates'
    var_3 = 'template content'
    var_4 = 'existing content'
    assert var_4 == 'existing content'
    var_5 = 'is_binary'
    var_6 = False
    var_7 = module_0.Environment()
    var_8 = 'cookiecutter'
    var_9 = {}
    var_10 = {var_8: var_9}
    var_11 = True

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = '{{ cookiecutter.dir_name }}/file.txt'
    var_2 = 'templates'
    var_3 = 'subdir'
    var_4 = 'content'
    var_5 = 'is_binary'
    var_6 = False
    var_7 = 'os.path.isdir'
    var_8 = True
    var_9 = module_0.Environment()
    var_10 = 'cookiecutter'
    var_11 = 'dir_name'
    var_12 = ''
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'file_with_custom_newline.txt'
    var_2 = 'templates'
    var_3 = 'line1\r\nline2\r\n'
    var_4 = 'is_binary'
    var_5 = False
    var_6 = module_0.Environment()
    var_7 = 'cookiecutter'
    var_8 = '_new_lines'
    var_9 = '\r\n'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'file_with_newline.txt'
    var_2 = 'templates'
    var_3 = 'content\n'
    var_4 = 'is_binary'
    var_5 = False
    var_6 = module_0.Environment()
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'bad_template.txt'
    var_2 = 'templates'
    var_3 = '{% if unclosed %}'
    var_4 = 'is_binary'
    var_5 = False
    var_6 = module_0.Environment()
    var_7 = 'cookiecutter'
    var_8 = {}
    var_9 = {var_7: var_8}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_59_evaluates_to_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 59 (with work_in(template_dir):) evaluates to False.'



# Parsed testcases at query #29
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new_var'
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
    var_7 = [var_2, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'provided for multi-choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = var_5['choice'][0]
    assert var_8 == 'b'
    var_9 = 'a'
    var_10 = bool('a' in var_5['choice'])
    assert var_10 is True
    var_11 = 'c'
    var_12 = bool('c' in var_5['choice'])
    assert var_12 is True

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
    var_6 = var_2['bool_var']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'no'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['bool_var']
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'true'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['bool_var']
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'false'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['bool_var']
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'maybe'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'could not be converted to a boolean'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'name': 'new'})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'count'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = 10
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'count': 10})
    assert var_6 is True

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
    var_0 = 'nested'
    var_1 = 'choices'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_3, var_4]
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = True
    var_12 = module_0.apply_overwrites_to_context(var_7, var_10, in_dictionary_variable=var_11)
    var_13 = var_7['nested']['choices']
    var_14 = bool(var_7['nested']['choices'] == ['b', 'c'])
    assert var_14 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'items'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'x'
    var_8 = 'y'
    var_9 = [var_7, var_8]
    var_10 = {var_1: var_9}
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_0.apply_overwrites_to_context(var_6, var_11, in_dictionary_variable=var_12)
    var_14 = var_6['nested']['items']
    var_15 = bool(var_6['nested']['items'] == ['x', 'y'])
    assert var_15 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_generate_context_predicate_line_18_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = "Test that the predicate at line 18 (open() call) evaluates to False when file doesn't exist."
    var_1 = 'non_existent_cookiecutter.json'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_generate_files_with_minimal_context. Retrieved 20/35 statements.
# Partially parsed test_generate_files_with_default_context. Retrieved 16/31 statements.
# Partially parsed test_generate_files_with_hooks_disabled. Retrieved 21/37 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 20/35 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 20/35 statements.
# Partially parsed test_generate_files_keeps_project_on_failure. Retrieved 20/35 statements.
# Partially parsed test_generate_files_with_custom_output_dir. Retrieved 14/27 statements.


def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'output'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.create_env_with_context'
    var_10 = 'cookiecutter.generate.render_and_create_dir'
    var_11 = True
    var_12 = 'cookiecutter.generate.work_in'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = 'os.walk'
    var_15 = '.'
    var_16 = []
    var_17 = []
    var_18 = (var_15, var_16, var_17)
    var_19 = [var_18]

def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'output'
    var_3 = 'cookiecutter.generate.find_template'
    var_4 = 'cookiecutter.generate.create_env_with_context'
    var_5 = 'cookiecutter.generate.render_and_create_dir'
    var_6 = 'project'
    var_7 = True
    var_8 = 'cookiecutter.generate.work_in'
    var_9 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_10 = 'os.walk'
    var_11 = '.'
    var_12 = []
    var_13 = []
    var_14 = (var_11, var_12, var_13)
    var_15 = [var_14]

def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'output'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.create_env_with_context'
    var_10 = 'cookiecutter.generate.render_and_create_dir'
    var_11 = True
    var_12 = 'cookiecutter.generate.work_in'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = 'os.walk'
    var_15 = '.'
    var_16 = []
    var_17 = []
    var_18 = (var_15, var_16, var_17)
    var_19 = [var_18]
    var_20 = False

def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'output'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.create_env_with_context'
    var_10 = 'cookiecutter.generate.render_and_create_dir'
    var_11 = True
    var_12 = 'cookiecutter.generate.work_in'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = 'os.walk'
    var_15 = '.'
    var_16 = []
    var_17 = []
    var_18 = (var_15, var_16, var_17)
    var_19 = [var_18]

def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'output'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.create_env_with_context'
    var_10 = 'cookiecutter.generate.render_and_create_dir'
    var_11 = True
    var_12 = 'cookiecutter.generate.work_in'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = 'os.walk'
    var_15 = '.'
    var_16 = []
    var_17 = []
    var_18 = (var_15, var_16, var_17)
    var_19 = [var_18]

def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'output'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.create_env_with_context'
    var_10 = 'cookiecutter.generate.render_and_create_dir'
    var_11 = True
    var_12 = 'cookiecutter.generate.work_in'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_14 = 'os.walk'
    var_15 = '.'
    var_16 = []
    var_17 = []
    var_18 = (var_15, var_16, var_17)
    var_19 = [var_18]

def test_case_0():
    var_0 = 'repo'
    var_1 = '{{cookiecutter.project_name}}'
    var_2 = 'custom_output'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.create_env_with_context'
    var_10 = 'cookiecutter.generate.render_and_create_dir'
    var_11 = True
    var_12 = 'cookiecutter.generate.work_in'
    var_13 = 'cookiecutter.generate.run_hook_from_repo_dir'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/8 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/11 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/11 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/11 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 6/11 statements.
# Partially parsed test_generate_context_with_dict_variable. Retrieved 8/13 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/9 statements.
# Partially parsed test_generate_context_missing_file. Retrieved 2/7 statements.
# Partially parsed test_generate_context_invalid_choice. Retrieved 7/12 statements.
# Partially parsed test_generate_context_invalid_boolean_conversion. Retrieved 7/12 statements.
# Partially parsed test_generate_context_multichoice_variable. Retrieved 9/16 statements.
# Partially parsed test_generate_context_custom_filename. Retrieved 3/8 statements.
# Partially parsed test_generate_context_with_default_and_extra_context. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'Test generate_context with a basic JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'cookiecutter'

def test_case_0():
    var_0 = 'Test generate_context with default_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'author'
    var_4 = 'Jane'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with extra_context parameter.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "author": "John"}'
    var_3 = 'project_name'
    var_4 = 'new_project'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with choice variable and extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}
    var_6 = 'MIT'

def test_case_0():
    var_0 = 'Test generate_context with boolean variable and extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'false'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"config": {"key1": "value1", "key2": "value2"}}'
    var_3 = 'config'
    var_4 = 'key1'
    var_5 = 'new_value1'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"invalid": json}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'JSON decoding error'

def test_case_0():
    var_0 = 'Test generate_context with missing context file.'
    var_1 = 'nonexistent.json'
    var_2 = bool(False)
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid choice value.'
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
    var_0 = 'Test generate_context with invalid boolean conversion.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'invalid_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'could not be converted to a boolean'

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
    var_0 = 'Test generate_context with custom context file name.'
    var_1 = 'custom_context.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = 'custom_context'

def test_case_0():
    var_0 = 'Test generate_context with both default_context and extra_context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"name": "project", "author": "John", "license": "MIT"}'
    var_3 = 'author'
    var_4 = 'Jane'
    var_5 = {var_3: var_4}
    var_6 = 'license'
    var_7 = 'Apache'
    var_8 = {var_6: var_7}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 12/29 statements.
# Partially parsed test_generate_files_with_subdirectories. Retrieved 13/32 statements.
# Partially parsed test_generate_files_skip_if_exists. Retrieved 14/35 statements.
# Partially parsed test_generate_files_empty_context. Retrieved 7/22 statements.
# Partially parsed test_generate_files_overwrite_if_exists. Retrieved 14/34 statements.
# Partially parsed test_generate_files_binary_file. Retrieved 12/30 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 13/30 statements.


def test_case_0():
    var_0 = 'Test generate_files creates project from template.'
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
    var_13 = 'my_project'

def test_case_0():
    var_0 = 'Test generate_files handles subdirectories correctly.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'src'
    var_4 = 'main.py'
    var_5 = '# {{cookiecutter.name}}'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'myapp'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = 'output'

def test_case_0():
    var_0 = 'Test generate_files skips files when skip_if_file_exists is True.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.proj}}'
    var_3 = 'config.txt'
    var_4 = 'config={{cookiecutter.proj}}'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'proj'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False
    var_14 = True

def test_case_0():
    var_0 = 'Test generate_files with empty context uses default.'
    var_1 = 'repo'
    var_2 = 'static_name'
    var_3 = 'file.txt'
    var_4 = 'content'
    var_5 = 'output'
    var_6 = None

def test_case_0():
    var_0 = 'Test generate_files can overwrite existing output directory.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'file.txt'
    var_4 = 'v1'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'project'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False
    var_14 = True

def test_case_0():
    var_0 = 'Test generate_files handles binary files correctly.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'image.bin'
    var_4 = b'\x89PNG\r\n\x1a\n'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = 'app'
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = [var_9]
    var_11 = [var_10]
    var_12 = 'output'

def test_case_0():
    var_0 = 'Test generate_files with accept_hooks=False.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.name}}'
    var_3 = 'test.txt'
    var_4 = 'test'
    var_5 = 'output'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = 'proj'
    var_9 = {var_7: var_8}
    var_10 = (var_6, var_9)
    var_11 = [var_10]
    var_12 = [var_11]
    var_13 = False

def test_case_0():
    var_0 = 'Test generate_files handles multiple files correctly.'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_generate_files_with_context_and_default_output_dir. Retrieved 28/44 statements.
# Partially parsed test_generate_files_without_context. Retrieved 16/29 statements.
# Partially parsed test_generate_files_with_overwrite_if_exists. Retrieved 18/30 statements.
# Partially parsed test_generate_files_with_skip_if_file_exists. Retrieved 22/36 statements.
# Partially parsed test_generate_files_without_hooks. Retrieved 19/31 statements.
# Partially parsed test_generate_files_with_undefined_error. Retrieved 12/23 statements.
# Partially parsed test_generate_files_keep_project_on_failure. Retrieved 18/30 statements.
# Partially parsed test_generate_files_with_copy_only_dirs. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Test generate_files with context and default output directory.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'test.txt'
    var_4 = 'Hello {{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'cookiecutter.generate.find_template'
    var_11 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_12 = 'cookiecutter.generate.os.walk'
    var_13 = '.'
    var_14 = 'subdir'
    var_15 = [var_14]
    var_16 = [var_3]
    var_17 = (var_13, var_15, var_16)
    var_18 = './subdir'
    var_19 = []
    var_20 = []
    var_21 = (var_18, var_19, var_20)
    var_22 = [var_17, var_21]
    var_23 = 'cookiecutter.generate.is_copy_only_path'
    var_24 = False
    var_25 = 'cookiecutter.generate.generate_file'
    var_26 = 'cookiecutter.generate.render_and_create_dir'
    var_27 = True

def test_case_0():
    var_0 = 'Test generate_files without context uses empty OrderedDict.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter.generate.find_template'
    var_4 = 'cookiecutter.generate.create_env_with_context'
    var_5 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_6 = 'cookiecutter.generate.os.walk'
    var_7 = '.'
    var_8 = []
    var_9 = []
    var_10 = (var_7, var_8, var_9)
    var_11 = [var_10]
    var_12 = 'cookiecutter.generate.render_and_create_dir'
    var_13 = 'project'
    var_14 = True
    var_15 = None

def test_case_0():
    var_0 = 'Test generate_files with overwrite_if_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_10 = 'cookiecutter.generate.os.walk'
    var_11 = '.'
    var_12 = []
    var_13 = []
    var_14 = (var_11, var_12, var_13)
    var_15 = [var_14]
    var_16 = 'cookiecutter.generate.render_and_create_dir'
    var_17 = True

def test_case_0():
    var_0 = 'Test generate_files with skip_if_file_exists flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_10 = 'cookiecutter.generate.os.walk'
    var_11 = '.'
    var_12 = []
    var_13 = 'file.txt'
    var_14 = [var_13]
    var_15 = (var_11, var_12, var_14)
    var_16 = [var_15]
    var_17 = 'cookiecutter.generate.is_copy_only_path'
    var_18 = False
    var_19 = 'cookiecutter.generate.render_and_create_dir'
    var_20 = True
    var_21 = 'cookiecutter.generate.generate_file'

def test_case_0():
    var_0 = 'Test generate_files with accept_hooks set to False.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_9 = 'cookiecutter.generate.find_template'
    var_10 = 'cookiecutter.generate.os.walk'
    var_11 = '.'
    var_12 = []
    var_13 = []
    var_14 = (var_11, var_12, var_13)
    var_15 = [var_14]
    var_16 = 'cookiecutter.generate.render_and_create_dir'
    var_17 = True
    var_18 = False

import jinja2.exceptions as module_0

def test_case_0():
    var_0 = 'Test generate_files raises UndefinedVariableInTemplate on UndefinedError.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.render_and_create_dir'
    var_10 = 'undefined'
    var_11 = module_0.UndefinedError(var_10)
    var_12 = bool(False)
    assert var_12 is True

def test_case_0():
    var_0 = 'Test generate_files with keep_project_on_failure flag.'
    var_1 = 'repo'
    var_2 = '{{cookiecutter.project_name}}'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter.generate.find_template'
    var_9 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_10 = 'cookiecutter.generate.os.walk'
    var_11 = '.'
    var_12 = []
    var_13 = []
    var_14 = (var_11, var_12, var_13)
    var_15 = [var_14]
    var_16 = 'cookiecutter.generate.render_and_create_dir'
    var_17 = True

def test_case_0():
    var_0 = 'Test generate_files with copy_only directories.'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_generate_context_basic. Retrieved 3/7 statements.
# Partially parsed test_generate_context_with_default_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_choice_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_boolean_variable. Retrieved 6/10 statements.
# Partially parsed test_generate_context_with_dict_variable. Retrieved 8/12 statements.
# Partially parsed test_generate_context_invalid_json. Retrieved 4/8 statements.
# Partially parsed test_generate_context_invalid_choice_override. Retrieved 7/11 statements.
# Partially parsed test_generate_context_invalid_multichoice_override. Retrieved 9/13 statements.
# Partially parsed test_generate_context_with_multichoice_valid. Retrieved 8/12 statements.
# Partially parsed test_generate_context_boolean_yes. Retrieved 6/10 statements.
# Partially parsed test_generate_context_boolean_no. Retrieved 6/10 statements.
# Partially parsed test_generate_context_boolean_true_string. Retrieved 6/10 statements.
# Partially parsed test_generate_context_invalid_boolean_string. Retrieved 7/11 statements.
# Partially parsed test_generate_context_with_both_contexts. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 'Test generate_context with a basic JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "my_project", "project_slug": "my_project"}'
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
    var_0 = 'Test generate_context with choice variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache", "GPL"]}'
    var_3 = 'license'
    var_4 = 'Apache'
    var_5 = {var_3: var_4}
    var_6 = 'MIT'
    var_7 = 'GPL'

def test_case_0():
    var_0 = 'Test generate_context with boolean variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_docker": true}'
    var_3 = 'use_docker'
    var_4 = 'false'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context with nested dictionary variable.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"author": {"name": "John", "email": "john@example.com"}}'
    var_3 = 'author'
    var_4 = 'name'
    var_5 = 'Jane'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid JSON file.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"invalid json}'
    var_3 = module_0.generate_context(var_0)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'JSON decoding error'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid choice override.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"license": ["MIT", "Apache"]}'
    var_3 = 'license'
    var_4 = 'GPL'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'choice variable'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid multichoice override.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"licenses": ["MIT", "Apache"]}'
    var_3 = 'licenses'
    var_4 = 'MIT'
    var_5 = 'GPL'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'multi-choice variable'

def test_case_0():
    var_0 = 'Test generate_context with valid multichoice override.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"licenses": ["MIT", "Apache", "GPL"]}'
    var_3 = 'licenses'
    var_4 = 'Apache'
    var_5 = 'GPL'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

def test_case_0():
    var_0 = 'Test generate_context converting yes string to boolean.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_feature": false}'
    var_3 = 'use_feature'
    var_4 = 'yes'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'Test generate_context converting no string to boolean.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"use_feature": true}'
    var_3 = 'use_feature'
    var_4 = 'no'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = "Test generate_context converting 'true' string to boolean."
    var_1 = 'cookiecutter.json'
    var_2 = '{"debug": false}'
    var_3 = 'debug'
    var_4 = 'true'
    var_5 = {var_3: var_4}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test generate_context with invalid boolean string.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"debug": false}'
    var_3 = 'debug'
    var_4 = 'maybe'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'could not be converted to a boolean'

def test_case_0():
    var_0 = 'Test generate_context with both default and extra context.'
    var_1 = 'cookiecutter.json'
    var_2 = '{"name": "original", "version": "1.0", "author": "John"}'
    var_3 = 'version'
    var_4 = '2.0'
    var_5 = {var_3: var_4}
    var_6 = 'author'
    var_7 = 'Jane'
    var_8 = {var_6: var_7}



