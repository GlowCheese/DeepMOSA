####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = {}
    var_4 = False
    var_5 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = bool(var_2 == {'existing': 'value'})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = 'new_nested'
    var_7 = 'new_value'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = True
    var_11 = module_0.apply_overwrites_to_context(var_4, var_9, in_dictionary_variable=var_10)
    var_12 = bool(var_4 == {'existing': {'nested': 'value'}, 'new': {'new_nested': 'new_value'}})
    assert var_12 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'choices': ['b', 'a', 'c']})
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
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'multichoices': ['a', 'c']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoices'
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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nested'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_value2'
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'nested': {'key1': 'value1', 'key2': 'new_value2'}})
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
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

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



# Parsed testcases at query #3
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'c'
    var_6 = 'd'
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)
    var_11 = var_4['key']
    var_12 = bool(var_4['key'] == ['c', 'd'])
    assert var_12 is True



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'variable'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_render_and_create_dir_with_existing_directory_and_no_overwrite. Retrieved 7/10 statements.
# Partially parsed test_render_and_create_dir_with_existing_directory_and_overwrite. Retrieved 7/9 statements.
# Partially parsed test_render_and_create_dir_with_new_directory. Retrieved 7/11 statements.
# Partially parsed test_render_and_create_dir_with_rendered_name. Retrieved 9/13 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'existing_dir'
    var_4 = {}
    var_5 = '/tmp'
    var_6 = module_0.Environment()
    var_7 = module_1.render_and_create_dir(var_3, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'existing_dir'
    var_4 = {}
    var_5 = '/tmp'
    var_6 = module_0.Environment()
    var_7 = module_1.render_and_create_dir(var_3, var_4, var_5, var_6, var_2)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'new_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = '/tmp/new_dir'
    var_6 = [var_5]
    var_7 = True
    var_8 = [var_5]

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '{{ name }}_dir'
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_3, var_2, var_4, var_5)
    var_7 = '/tmp/test_dir'
    var_8 = [var_7]
    var_9 = True
    var_10 = [var_7]



# Parsed testcases at query #6
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = '*.md'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'README.md'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'notes.txt'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = '*.md'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'main.py'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False
    var_9 = 'data.json'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'README.md'
    var_6 = module_0.is_copy_only_path(var_5, var_4)
    assert var_6 is False
    var_7 = 'main.py'
    var_8 = module_0.is_copy_only_path(var_7, var_4)
    assert var_8 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'README.md'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False
    var_5 = 'main.py'
    var_6 = module_0.is_copy_only_path(var_5, var_2)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'README.md'
    var_2 = module_0.is_copy_only_path(var_1, var_0)
    assert var_2 is False
    var_3 = 'main.py'
    var_4 = module_0.is_copy_only_path(var_3, var_0)
    assert var_4 is False



# Parsed testcases at query #7
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test-cookiecutter': {'name': 'test'}})
    assert var_2 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_empty_dirname_raises_exception. Retrieved 3/6 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = module_0.Environment()



# Parsed testcases at query #9
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test_cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'test_cookiecutter': {'name': 'default', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'test_cookiecutter': {'name': 'extra', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



# Parsed testcases at query #11
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_bool'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #12
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_bool'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_generate_context_raises_exception_on_invalid_json. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = str(var_0)
    var_3 = 'JSON decoding error while loading'
    var_4 = bool('JSON decoding error while loading' in var_2)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #15
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'output_dir'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/17 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 8/16 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 10/20 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 9/18 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/18 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 7/13 statements.
# Partially parsed test_generate_files_keep_project_on_failure. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'basic-template'
    var_7 = 'README.md'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'basic-template'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'basic-template'
    var_7 = 'test'
    var_8 = True
    var_9 = 'new_file.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'template-with-hooks'
    var_7 = True
    var_8 = 'hook_output.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.bin'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'test-data'
    var_9 = 'template-with-binary'
    var_10 = 'data.bin'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'template-with-undefined'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'template-with-undefined'
    var_7 = True



# Parsed testcases at query #17
#--------------------------




import json as module_0

def test_case_0():
    var_0 = '{"invalid": json}'
    var_1 = {}
    var_2 = module_0.loads(var_0, **var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_generate_context_opens_file. Retrieved 3/6 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = '{"key": "value"}'
    var_2 = module_0.generate_context(var_0)
    var_3 = bool(var_2 == {'test_context': {'key': 'value'}})
    assert var_3 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_existing_dir_no_overwrite. Retrieved 3/6 statements.
# Partially parsed test_render_and_create_dir_existing_dir_overwrite. Retrieved 4/8 statements.
# Partially parsed test_render_and_create_dir_new_dir. Retrieved 3/6 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = []
    var_3 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = True
    var_3 = {}
    var_4 = []
    var_5 = module_0.Environment()

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = []
    var_3 = module_0.Environment()
    var_4 = [var_0]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_3: var_1}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'existing': 'value'})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'existing'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = 'new_nested'
    var_7 = 'new_value'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = True
    var_11 = module_0.apply_overwrites_to_context(var_4, var_9, in_dictionary_variable=var_10)
    var_12 = bool(var_4 == {'existing': {'nested': 'value'}, 'new': {'new_nested': 'new_value'}})
    assert var_12 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'list_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_1]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'list_var': ['b', 'a', 'c']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'list_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = [var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multi_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'multi_var': ['a', 'c']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multi_var'
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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'dict_var': {'a': 1, 'b': 3}})
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
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'var': 'new'})
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'invalid_value'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = var_4['key']
    var_9 = bool(var_4['key'] == ['value1', 'value2'])
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/17 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 9/18 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 8/16 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 10/22 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/20 statements.
# Partially parsed test_generate_files_undefined_variable_failure. Retrieved 7/13 statements.
# Partially parsed test_generate_files_keep_project_on_failure. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'basic-template'
    var_7 = 'README.md'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'template-with-hooks'
    var_7 = True
    var_8 = 'hook_output.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'basic-template'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'basic-template'
    var_7 = 'existing content'
    assert var_7 == 'existing content'
    var_8 = True
    var_9 = 'new_file.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    assert var_0 == b'binary content'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.bin'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'test-data'
    var_9 = 'template-with-binaries'
    var_10 = 'data.bin'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'template-with-undefined-variable'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'template-with-undefined-variable'
    var_7 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__run_hook_from_repo_dir_deprecation_warning. Retrieved 7/16 statements.
# Partially parsed test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir. Retrieved 7/9 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = 'repo_dir'
    var_2 = 'hook_name'
    var_3 = 'project_dir'
    var_4 = {}
    var_5 = False
    var_6 = module_0._run_hook_from_repo_dir(var_1, var_2, var_3, var_4, var_5)
    var_7 = "The '_run_hook_from_repo_dir' function is deprecated"

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = {}
    var_4 = False
    var_5 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)
    var_6 = {}



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = False
    var_2 = bool(not (var_1 and var_0))
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_apply_overwrites_to_context_boolean_invalid_response. Retrieved 6/9 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = 'invalid provided for variable test_var could not be converted to a boolean.'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 14/22 statements.
# Partially parsed test_generate_file_text_file. Retrieved 14/22 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 15/23 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 15/18 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 14/20 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 14/20 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = b'\x89PNG\r\n\x1a\n'
    assert var_11 == b'\x89PNG\r\n\x1a\n'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = module_3.rmtree(var_0)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello, {{ name }}!'
    assert var_11 == 'Hello, {{ name }}!'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = module_3.rmtree(var_0)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Original content'
    var_12 = 'Existing content'
    assert var_12 == 'Existing content'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_14 = module_3.rmtree(var_0)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = '{{ empty_var }}.txt'
    var_2 = 'cookiecutter'
    var_3 = 'empty_var'
    var_4 = '_new_lines'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = ''
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_14 = module_3.rmtree(var_0)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Line 1\r\nLine 2\r\n'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = b'\r\n'
    var_14 = module_3.rmtree(var_0)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'custom_newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Line 1\r\nLine 2\r\n'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = b'\r\n'
    var_14 = b'\n'
    var_15 = module_3.rmtree(var_0)



# Parsed testcases at query #8
#--------------------------




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



# Parsed testcases at query #9
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'test-cookiecutter': {'name': 'default', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'version'
    var_2 = '2.0.0'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'test-cookiecutter': {'name': 'test', 'version': '2.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'version'
    var_5 = '2.0.0'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, var_3, var_6)
    var_8 = bool(var_7 == {'test-cookiecutter': {'name': 'default', 'version': '2.0.0'}})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-nested-cookiecutter.json'
    var_1 = 'config'
    var_2 = 'debug'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(var_6 == {'test-nested-cookiecutter': {'name': 'test', 'config': {'debug': True, 'verbose': False}}})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-list-cookiecutter.json'
    var_1 = 'choices'
    var_2 = 'option2'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = module_0.generate_context(var_0, extra_context=var_4)
    var_6 = bool(var_5 == {'test-list-cookiecutter': {'choices': ['option2', 'option1', 'option3']}})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-list-cookiecutter.json'
    var_1 = 'choices'
    var_2 = 'invalid'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = module_0.generate_context(var_0, extra_context=var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-bool-cookiecutter.json'
    var_1 = 'flag'
    var_2 = 'yes'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'test-bool-cookiecutter': {'flag': True}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-bool-cookiecutter.json'
    var_1 = 'flag'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_render_and_create_dir_rendered_name. Retrieved 9/10 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'existing_dir'
    var_7 = {}
    var_8 = '/tmp'
    var_9 = module_1.Environment()
    var_10 = module_2.render_and_create_dir(var_6, var_7, var_8, var_9)
    var_11 = var_3.rmdir()

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'existing_dir'
    var_7 = {}
    var_8 = '/tmp'
    var_9 = module_1.Environment()
    var_10 = module_2.render_and_create_dir(var_6, var_7, var_8, var_9, var_4)
    var_11 = bool(var_10 == (var_3, False))
    assert var_11 is True
    var_12 = var_3.rmdir()

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/new_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'new_dir'
    var_5 = {}
    var_6 = '/tmp'
    var_7 = module_1.Environment()
    var_8 = module_2.render_and_create_dir(var_4, var_5, var_6, var_7)
    var_9 = bool(var_8 == (var_3, True))
    assert var_9 is True
    var_10 = var_3.rmdir()

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ name }}_dir'
    var_5 = '/tmp'
    var_6 = module_1.render_and_create_dir(var_4, var_2, var_5, var_3)
    var_7 = var_6[0].name
    assert var_7 == 'test_dir'
    var_8 = 0
    var_9 = var_6[var_8]



# Parsed testcases at query #11
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1
import pathlib as module_2

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_0, var_3, var_4, var_5)
    var_7 = [var_4, var_0]
    var_8 = {}
    var_9 = module_2.Path(*var_7, **var_8)
    var_10 = var_6[0]
    var_11 = bool(var_6[0] == var_9)
    assert var_11 is True
    var_12 = var_6[1]
    assert var_12 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_0, var_3, var_4, var_5)

import jinja2.environment as module_0
import pathlib as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = [var_4, var_0]
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = True
    var_10 = var_8.mkdir(exist_ok=var_9)
    var_11 = module_2.render_and_create_dir(var_0, var_3, var_4, var_5)

import jinja2.environment as module_0
import pathlib as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = [var_4, var_0]
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = True
    var_10 = var_8.mkdir(exist_ok=var_9)
    var_11 = module_2.render_and_create_dir(var_0, var_3, var_4, var_5, var_9)
    var_12 = var_11[0]
    var_13 = bool(var_11[0] == var_8)
    assert var_13 is True
    var_14 = var_11[1]
    assert var_14 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_template_syntax_error_raised_with_verbose_info. Retrieved 12/15 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'fake_template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = '{% if %}'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #13
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #14
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'x'
    var_7 = 'y'
    var_8 = [var_6, var_7]
    var_9 = {var_0: var_8}
    var_10 = True
    var_11 = module_0.apply_overwrites_to_context(var_5, var_9, in_dictionary_variable=var_10)
    var_12 = var_5['key']
    var_13 = bool(var_5['key'] == ['x', 'y'])
    assert var_13 is True



# Parsed testcases at query #15
#--------------------------




import cookiecutter.utils as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = module_0.create_env_with_context(var_0)
    var_3 = '{{ invalid_var }}'
    var_4 = False
    var_5 = module_1.render_and_create_dir(var_3, var_0, var_1, var_2, var_4)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_cookiecutter_new_lines_predicate_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #17
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #18
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #19
#--------------------------




import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'valid_template.txt'
    var_2 = var_0.get_template(var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_text_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/20 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 13/15 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = True
    var_12 = 'w'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{""}}'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = b'\r\n'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_new_lines_in_context. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #22
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #23
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 10/16 statements.
# Partially parsed test_generate_file_text_file. Retrieved 10/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 12/17 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 11/13 statements.
# Partially parsed test_generate_file_with_template_rendering. Retrieved 12/16 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = True
    var_10 = 'existing content'
    var_11 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{""}}'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'test'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generate_context_json_decoding_error. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = str(var_0)
    var_3 = 'JSON decoding error while loading'
    var_4 = bool('JSON decoding error while loading' in var_2)
    assert var_4 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_text_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/20 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 13/15 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template/dir'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template/dir'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template/dir'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = True
    var_12 = 'w'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = '{{""}}'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template/dir'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template/dir'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = b'\r\n'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #29
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #30
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'default', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'extra'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)
    var_7 = bool(var_6 == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_5 is True



