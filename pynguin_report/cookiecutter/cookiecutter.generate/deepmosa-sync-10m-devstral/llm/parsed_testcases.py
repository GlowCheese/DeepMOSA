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
    var_4 = 'overwrite'
    var_5 = {var_3: var_4}
    var_6 = module_0.apply_overwrites_to_context(var_2, var_5)
    var_7 = bool(var_2 == {'existing': 'value'})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'existing'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = 'overwrite'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = bool(var_4 == {'dict_var': {'existing': 'value', 'new': 'overwrite'}})
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'list_var'
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
    var_11 = bool(var_5 == {'list_var': ['x', 'y']})
    assert var_11 is True

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
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = "d provided for multi-choice variable multi_var, but valid choices are ['a', 'b', 'c']"

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = {var_0: var_2}
    var_7 = module_0.apply_overwrites_to_context(var_5, var_6)
    var_8 = bool(var_5 == {'choice_var': ['b', 'a', 'c']})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choice_var'
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
    var_10 = "d provided for choice variable choice_var, but the choices are ['a', 'b', 'c']."

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'd'
    var_10 = 20
    var_11 = 4
    var_12 = {var_2: var_10, var_9: var_11}
    var_13 = {var_0: var_12}
    var_14 = module_0.apply_overwrites_to_context(var_8, var_13)
    var_15 = bool(var_8 == {'dict_var': {'a': 1, 'b': 20, 'c': 3, 'd': 4}})
    assert var_15 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
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
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'invalid provided for variable bool_var could not be converted to a boolean.'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var'
    var_1 = 'original'
    var_2 = {var_0: var_1}
    var_3 = 'overwrite'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'var': 'overwrite'})
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir. Retrieved 9/11 statements.
# Partially parsed test__run_hook_from_repo_dir_emits_deprecation_warning. Retrieved 10/20 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)
    var_8 = {var_3: var_4}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = 'repo_dir'
    var_2 = 'hook_name'
    var_3 = 'project_dir'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = True
    var_8 = module_0._run_hook_from_repo_dir(var_1, var_2, var_3, var_6, var_7)
    var_9 = 0
    var_10 = "_run_hook_from_repo_dir' function is deprecated"



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/10 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 9/11 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 9/11 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 9/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 9/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = module_0.generate_files(var_5, var_4, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = False
    var_8 = module_0.generate_files(var_5, var_4, var_6, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, keep_project_on_failure=var_7)



# Parsed testcases at query #4
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_0, var_1, var_4, var_5)

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'existing_dir'
    var_7 = {}
    var_8 = []
    var_9 = {}
    var_10 = module_0.Path(*var_8, **var_9)
    var_11 = module_1.Environment()
    var_12 = module_2.render_and_create_dir(var_6, var_7, var_10, var_11)

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = {}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Path(*var_7, **var_8)
    var_10 = module_1.Environment()
    var_11 = module_2.render_and_create_dir(var_0, var_6, var_9, var_10, var_4)
    var_12 = bool(var_11 == (var_3, False))
    assert var_12 is True

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'new_dir'
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_0, var_1, var_4, var_5)
    var_7 = [var_0]
    var_8 = {}
    var_9 = module_0.Path(*var_7, **var_8)
    var_10 = True
    var_11 = (var_9, var_10)
    var_12 = bool(var_6 == var_11)
    assert var_12 is True
    var_13 = [var_0]
    var_14 = {}
    var_15 = module_0.Path(*var_13, **var_14)
    var_16 = var_15.rmdir()

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '{{ name }}_dir'
    var_4 = []
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = module_1.Environment()
    var_8 = module_2.render_and_create_dir(var_3, var_2, var_6, var_7)
    var_9 = 'test_dir'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Path(*var_10, **var_11)
    var_13 = True
    var_14 = (var_12, var_13)
    var_15 = bool(var_8 == var_14)
    assert var_15 is True
    var_16 = [var_9]
    var_17 = {}
    var_18 = module_0.Path(*var_16, **var_17)
    var_19 = var_18.rmdir()



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'variable'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #7
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = module_1.Environment()
    var_7 = [var_5, var_0]
    var_8 = {}
    var_9 = module_0.Path(*var_7, **var_8)
    var_10 = True
    var_11 = var_9.mkdir(exist_ok=var_10)
    var_12 = module_2.render_and_create_dir(var_0, var_1, var_5, var_6)
    var_13 = var_12[1]
    assert var_13 is False



# Parsed testcases at query #8
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = module_1.Environment()
    var_7 = True
    var_8 = [var_5, var_0]
    var_9 = {}
    var_10 = module_0.Path(*var_8, **var_9)
    var_11 = True
    var_12 = var_10.mkdir(exist_ok=var_11)
    var_13 = var_10.exists()
    assert var_13 is True



# Parsed testcases at query #9
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = module_1.Environment()
    var_7 = True
    var_8 = [var_5, var_0]
    var_9 = {}
    var_10 = module_0.Path(*var_8, **var_9)
    var_11 = True
    var_12 = var_10.mkdir(parents=var_11, exist_ok=var_11)
    var_13 = module_2.render_and_create_dir(var_0, var_1, var_5, var_6, var_7)
    var_14 = var_13[1]
    assert var_14 is False



# Parsed testcases at query #10
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'subkey'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'not_a_dict'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = bool(var_4 == {'key': {'subkey': 'value'}})
    assert var_8 is True



# Parsed testcases at query #11
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'override'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'override', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'version'
    var_2 = '2.0.0'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'test', 'version': '2.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/missing.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/empty.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'empty': {}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/nested.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'nested': {'a': {'b': 1}}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/list.json'
    var_1 = 'items'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(var_6 == {'list': {'items': ['a', 'b']}})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/list.json'
    var_1 = 'items'
    var_2 = 'invalid'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = module_0.generate_context(var_0, extra_context=var_4)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/bool.json'
    var_1 = 'flag'
    var_2 = 'yes'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'bool': {'flag': True}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/bool.json'
    var_1 = 'flag'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



# Parsed testcases at query #12
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #13
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
    var_0 = 'dict'
    var_1 = 'existing'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'new'
    var_6 = {var_5: var_2}
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = bool(var_4 == {'dict': {'existing': 'value', 'new': 'value'}})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = 5
    var_8 = 6
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_0: var_9}
    var_11 = True
    var_12 = module_0.apply_overwrites_to_context(var_5, var_10, in_dictionary_variable=var_11)
    var_13 = bool(var_5 == {'list': [4, 5, 6]})
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'choices': ['a', 'c', 'b']})
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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_0: var_7}
    var_9 = 'd'
    var_10 = 20
    var_11 = 40
    var_12 = {var_2: var_10, var_9: var_11}
    var_13 = {var_0: var_12}
    var_14 = module_0.apply_overwrites_to_context(var_8, var_13)
    var_15 = bool(var_8 == {'dict': {'a': 1, 'b': 20, 'c': 3, 'd': 40}})
    assert var_15 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'yes'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'bool': True})
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool'
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
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'key': 'new'})
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test_cookiecutter': {'name': 'test', 'value': 1}})
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
    var_5 = bool(var_4 == {'test_cookiecutter': {'name': 'default', 'value': 1}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'test_cookiecutter': {'name': 'extra', 'value': 1}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'extra'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)
    var_7 = bool(var_6 == {'test_cookiecutter': {'name': 'extra', 'value': 1}})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'test_cookiecutter': {'name': 'test', 'value': 1}})
    assert var_5 is True



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_0, var_1, var_4, var_5)



# Parsed testcases at query #17
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_2 is True

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
    var_1 = 'version'
    var_2 = '2.0.0'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'test', 'version': '2.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_generate_context_opens_file. Retrieved 5/8 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = bool(var_4 == {'test_context': {'key': 'value'}})
    assert var_5 is True



# Parsed testcases at query #20
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False

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
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = []
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.is_copy_only_path(var_0, var_5)
    assert var_6 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.py'
    var_4 = '*.txt'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.is_copy_only_path(var_0, var_7)
    assert var_8 is True



# Parsed testcases at query #21
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'temp*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'temp_file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_0, var_1, var_4, var_5)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/17 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 8/15 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 8/15 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 8/15 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = 'README.md'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'template_with_hooks'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'failing_template'
    var_7 = True



# Parsed testcases at query #24
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = True
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = module_1.generate_file(var_0, var_1, var_4, var_5, var_6)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = False
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 10/16 statements.
# Partially parsed test_generate_file_text_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 12/16 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 10/12 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary.bin'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'test'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = True
    var_10 = 'w'
    var_11 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '{{""}}'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'newlines.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = b'\r\n'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test_cookiecutter': {'name': 'test', 'value': 1}})
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
    var_5 = bool(var_4 == {'test_cookiecutter': {'name': 'default', 'value': 1}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'test_cookiecutter': {'name': 'extra', 'value': 1}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'extra'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)
    var_7 = bool(var_6 == {'test_cookiecutter': {'name': 'extra', 'value': 1}})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'test_cookiecutter': {'name': 'test', 'value': 1}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'cookiecutter': {'project_name': 'test_project', 'author': 'test_author'}})
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir. Retrieved 5/14 statements.
# Partially parsed test_render_and_create_dir_empty_dirname. Retrieved 3/10 statements.
# Partially parsed test_render_and_create_dir_exists_no_overwrite. Retrieved 5/14 statements.
# Partially parsed test_render_and_create_dir_exists_with_overwrite. Retrieved 6/17 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ project_name }}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Environment()
    var_2 = ''

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ project_name }}'

import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ project_name }}'
    var_5 = True



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'file.py'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is False

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
    var_0 = 'src/utils'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = 'src/*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = {var_1: var_5}
    var_7 = module_0.is_copy_only_path(var_0, var_6)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}})
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
    var_0 = 'nonexistent.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #6
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
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = bool(var_4 == {'existing': {'nested': 'value', 'new': 'new_value'}})
    assert var_10 is True

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
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_val1'
    var_8 = {var_1: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'dict_var': {'key1': 'new_val1', 'key2': 'val2'}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
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
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)

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



# Parsed testcases at query #7
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'project_slug'
    var_5 = 'test_slug'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, var_3, var_6)
    var_8 = var_7['cookiecutter']['project_name']
    assert var_8 == 'test_project'
    var_9 = var_7['cookiecutter']['project_slug']
    assert var_9 == 'test_slug'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/nonexistent.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = var_1['cookiecutter']['project_name']
    assert var_2 == 'default_project'
    var_3 = var_1['cookiecutter']['project_slug']
    assert var_3 == 'default_slug'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'use_pytest'
    var_2 = 'yes'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = var_4['cookiecutter']['use_pytest']
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'use_pytest'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'framework'
    var_2 = 'flask'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = var_4['cookiecutter']['framework']
    var_6 = bool(var_4['cookiecutter']['framework'] == ['flask', 'django', 'pyramid'])
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'framework'
    var_2 = 'invalid_framework'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'database'
    var_2 = 'name'
    var_3 = 'postgres'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = var_6['cookiecutter']['database']['name']
    assert var_7 == 'postgres'
    var_8 = var_6['cookiecutter']['database']['host']
    assert var_8 == 'localhost'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'new_variable'
    var_2 = 'new_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = 'new_variable'
    var_6 = bool('new_variable' not in var_4['cookiecutter'])
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = 'new_dict'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = var_6['cookiecutter']['new_dict']
    var_8 = bool(var_6['cookiecutter']['new_dict'] == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #8
#--------------------------




import jinja2.environment as module_0
import pathlib as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = False
    var_5 = [var_2, var_0]
    var_6 = {}
    var_7 = module_1.Path(*var_5, **var_6)
    var_8 = True
    var_9 = var_7.mkdir(parents=var_8, exist_ok=var_8)
    var_10 = module_2.render_and_create_dir(var_0, var_1, var_2, var_3, var_4)
    var_11 = var_10[1]
    assert var_11 is False



# Parsed testcases at query #9
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['key']
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'default', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = var_4[1]
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = 'e'
    var_8 = [var_6, var_7]
    var_9 = {var_0: var_8}
    var_10 = True
    var_11 = module_0.apply_overwrites_to_context(var_5, var_9, in_dictionary_variable=var_10)
    var_12 = var_5['key']
    var_13 = bool(var_5['key'] == ['d', 'e'])
    assert var_13 is True



# Parsed testcases at query #13
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generate_context_opens_file. Retrieved 5/8 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = bool(var_4 == {'test_context': {'key': 'value'}})
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__run_hook_from_repo_dir_calls_run_hook_from_repo_dir. Retrieved 7/9 statements.
# Partially parsed test__run_hook_from_repo_dir_emits_deprecation_warning. Retrieved 8/10 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'repo'
    var_1 = 'hook'
    var_2 = 'project'
    var_3 = {}
    var_4 = False
    var_5 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)
    var_6 = {}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'repo'
    var_1 = 'hook'
    var_2 = 'project'
    var_3 = {}
    var_4 = False
    var_5 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_3, var_4)
    var_6 = "The '_run_hook_from_repo_dir' function is deprecated, use 'cookiecutter.hooks.run_hook_from_repo_dir' instead"
    var_7 = 2



# Parsed testcases at query #16
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #17
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'cookiecutter': {'project_name': 'My Project', 'project_slug': 'my_project', 'author': 'Your Name', 'email': 'your@email.com', 'version': '0.1.0'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'New Project'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = var_4['cookiecutter']['project_name']
    assert var_5 == 'New Project'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'Extra Project'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = var_4['cookiecutter']['project_name']
    assert var_5 == 'Extra Project'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'invalid_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #18
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'variable'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #19
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_dir'
    var_5 = {}
    var_6 = module_1.Environment()
    var_7 = [var_3, var_4]
    var_8 = {}
    var_9 = module_0.Path(*var_7, **var_8)
    var_10 = True
    var_11 = var_9.mkdir(exist_ok=var_10)
    var_12 = var_9.exists()
    var_13 = bool(var_12)
    assert var_13 is True



# Parsed testcases at query #20
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_0, var_1, var_4, var_5)



# Parsed testcases at query #21
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/17 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 8/14 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 9/19 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 9/16 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/18 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 7/13 statements.


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
    var_7 = 'existing content'
    assert var_7 == 'existing content'
    var_8 = True

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
    var_9 = 'template-with-binaries'
    var_10 = 'data.bin'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'template-with-undefined'



# Parsed testcases at query #23
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #24
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #25
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_0, var_3, var_4, var_5)
    var_7 = var_6[0].name
    assert var_7 == 'test_dir'
    var_8 = var_6[1]
    assert var_8 is True

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
    var_6 = 'test_dir'
    var_7 = [var_4, var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = True
    var_11 = var_9.mkdir(exist_ok=var_10)
    var_12 = module_2.render_and_create_dir(var_0, var_3, var_4, var_5)

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
    var_6 = 'test_dir'
    var_7 = [var_4, var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = True
    var_11 = var_9.mkdir(exist_ok=var_10)
    var_12 = module_2.render_and_create_dir(var_0, var_3, var_4, var_5, var_10)
    var_13 = var_12[0].name
    assert var_13 == 'test_dir'
    var_14 = var_12[1]
    assert var_14 is False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #27
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
    var_0 = 'multichoice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'multichoice': ['a', 'c']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice'
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
    var_11 = bool(var_6 == {'nested': {'a': 1, 'b': 3}})
    assert var_11 is True

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
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'old'
    var_2 = {var_0: var_1}
    var_3 = 'new'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(var_2 == {'key': 'new'})
    assert var_6 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_render_and_create_dir_new_dir. Retrieved 7/8 statements.


import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_0, var_1, var_4, var_5)

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'test_dir'
    var_7 = {}
    var_8 = module_1.Environment()
    var_9 = module_2.render_and_create_dir(var_6, var_7, var_3, var_8)

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'test_dir'
    var_7 = {}
    var_8 = module_1.Environment()
    var_9 = module_2.render_and_create_dir(var_6, var_7, var_3, var_8, var_4)
    var_10 = var_9[0]
    var_11 = bool(var_9[0] == var_3 / 'test_dir')
    assert var_11 is True
    var_12 = bool(not var_9[1])
    assert var_12 is True

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'new_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_dir'
    var_5 = {}
    var_6 = module_1.Environment()
    var_7 = module_2.render_and_create_dir(var_4, var_5, var_3, var_6)
    var_8 = var_7[0]
    var_9 = bool(var_7[0] == var_3 / 'test_dir')
    assert var_9 is True
    var_10 = bool(var_7[1])
    assert var_10 is True
    var_11 = var_3 / var_4



# Parsed testcases at query #29
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_var'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/17 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 8/16 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 9/19 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/20 statements.
# Partially parsed test_generate_files_hooks. Retrieved 9/16 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 7/13 statements.
# Partially parsed test_generate_files_keep_project_on_failure. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = 'file.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = 'existing content'
    assert var_7 == 'existing content'
    var_8 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.md'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'test_data'
    var_9 = 'copy_template'
    var_10 = 'readme.md'
    var_11 = '{{'
    var_12 = bool('{{' not in var_0)
    assert var_12 is True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'hook_template'
    var_7 = True
    var_8 = 'hook_output.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'undefined_template'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'undefined_template'
    var_7 = True



# Parsed testcases at query #31
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = None
    var_2 = module_0.generate_context(var_0, var_1, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_text_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/20 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 13/15 statements.
# Partially parsed test_generate_file_with_newline_config. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
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
    var_0 = '/tmp/test_project'
    var_1 = '{{""}}'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = b'\n'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 12/15 statements.
# Partially parsed test_generate_file_text_file. Retrieved 11/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = f'{var_0}/{var_1}'
    var_11 = f'{var_0}/{var_1}'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = f'{var_0}/{var_1}'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_file_name_is_empty_when_outfile_is_directory. Retrieved 9/12 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{cookiecutter.fake_var}}'
    var_2 = 'cookiecutter'
    var_3 = 'fake_var'
    var_4 = 'fake_dir'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 13/20 statements.
# Partially parsed test_generate_file_text_file. Retrieved 17/24 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 13/18 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 14/17 statements.
# Partially parsed test_generate_file_newline_config. Retrieved 13/18 statements.


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
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = b'\x00\x01\x02\x03'
    assert var_11 == b'\x00\x01\x02\x03'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

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
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello, {{ name }}!'
    assert var_11 == 'Hello, World!'
    var_12 = 'name'
    var_13 = 'World'
    var_14 = {var_3: var_4}
    var_15 = {var_12: var_13, var_11: var_14}
    var_16 = module_2.generate_file(var_0, var_1, var_15, var_9)

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
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Existing content'
    assert var_11 == 'Existing content'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = '{{ empty_var }}/file.txt'
    var_2 = 'empty_var'
    var_3 = 'cookiecutter'
    var_4 = ''
    var_5 = '_new_lines'
    var_6 = None
    var_7 = {var_5: var_6}
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = 'templates'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = module_2.generate_file(var_0, var_1, var_8, var_11)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Line 1\nLine 2'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = b'\r\n'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 11/18 statements.
# Partially parsed test_generate_file_text_rendering. Retrieved 13/20 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 11/17 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 12/15 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 11/16 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = b'\x00\x01\x02\x03'
    assert var_9 == b'\x00\x01\x02\x03'
    var_10 = module_2.generate_file(var_0, var_1, var_4, var_7)

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
    var_10 = True
    var_11 = 'Hello {{ cookiecutter.name }}!'
    assert var_11 == 'Hello test!'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = 'existing content'
    assert var_9 == 'existing content'
    var_10 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{ cookiecutter.empty }}'
    var_2 = 'cookiecutter'
    var_3 = 'empty'
    var_4 = ''
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'newlines.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = 'Line 1\r\nLine 2'
    var_10 = module_2.generate_file(var_0, var_1, var_4, var_7)
    var_11 = b'\r\n'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 13/20 statements.
# Partially parsed test_generate_file_text. Retrieved 15/22 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 13/19 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 14/17 statements.
# Partially parsed test_generate_file_newline_config. Retrieved 15/20 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = b'\x00\x01\x02'
    assert var_11 == b'\x00\x01\x02'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = None
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello {{ cookiecutter.name }}!'
    assert var_13 == 'Hello test!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'existing content'
    assert var_11 == 'existing content'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{ cookiecutter.empty }}.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'empty'
    var_5 = None
    var_6 = ''
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = module_2.generate_file(var_0, var_1, var_8, var_11)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'newline.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = '\r\n'
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello {{ cookiecutter.name }}!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_15 = b'\r\n'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 11/16 statements.
# Partially parsed test_generate_file_text_file. Retrieved 15/19 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'Hello {{ name }}'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_12 = 'w'
    var_13 = 'utf-8'
    var_14 = '\n'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 11/13 statements.
# Partially parsed test_generate_file_text_file. Retrieved 11/13 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 13/19 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 11/13 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 12/14 statements.
# Partially parsed test_generate_file_configured_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = 'binary.jpg'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = 'template.txt'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = True
    var_10 = 'existing.txt'
    var_11 = 'w'
    var_12 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = ''
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
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = b'\r\n'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #42
#--------------------------




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
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #43
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 12/16 statements.
# Partially parsed test_generate_file_text_rendering. Retrieved 14/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/20 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 13/15 statements.
# Partially parsed test_generate_file_newline_handling. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = 'test'
    var_6 = None
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = False
    var_13 = module_2.generate_file(var_0, var_1, var_8, var_11, var_12)
    var_14 = 'test'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
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
    var_0 = '/tmp/project'
    var_1 = '{{""}}'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'newline_test.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = b'\r\n'



# Parsed testcases at query #45
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'test_binary_file.bin'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 13/20 statements.
# Partially parsed test_generate_file_text. Retrieved 15/22 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 15/21 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 14/17 statements.
# Partially parsed test_generate_file_newline_config. Retrieved 15/20 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = b'\x00\x01\x02'
    assert var_11 == b'\x00\x01\x02'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = False
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'templates'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello {{ cookiecutter.name }}!'
    assert var_13 == 'Hello test!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = False
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'templates'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Existing content'
    assert var_13 == 'Existing content'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11, var_12)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{ cookiecutter.name }}.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = False
    var_6 = ''
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'templates'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = module_2.generate_file(var_0, var_1, var_8, var_11)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = '\r\n'
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'templates'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello {{ cookiecutter.name }}!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_15 = b'\r\n'
    var_16 = bool(b'\r\n' in var_13)
    assert var_16 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 12/17 statements.
# Partially parsed test_generate_file_text_file. Retrieved 15/24 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 12/15 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 12/15 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

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
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

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
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'w'
    var_13 = 'utf-8'
    var_14 = '\n'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

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
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = bool(not var_4)
    assert var_12 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = bool(not var_4)
    assert var_12 is True



# Parsed testcases at query #48
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = var_1['test-cookiecutter']['project_name']
    assert var_2 == 'Test Project'
    var_3 = var_1['test-cookiecutter']['author']
    assert var_3 == 'Test Author'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'Default Project'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test-cookiecutter.json'
    var_4 = module_0.generate_context(var_3, var_2)
    var_5 = var_4['test-cookiecutter']['project_name']
    assert var_5 == 'Default Project'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'Extra Project'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test-cookiecutter.json'
    var_4 = module_0.generate_context(var_3, extra_context=var_2)
    var_5 = var_4['test-cookiecutter']['project_name']
    assert var_5 == 'Extra Project'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid_var'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'tests/test-cookiecutter.json'
    var_4 = module_0.generate_context(var_3, var_2)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'non_existent.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_cookiecutter_new_lines_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_generate_file_binary_copy. Retrieved 10/13 statements.
# Partially parsed test_generate_file_text_render. Retrieved 16/22 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 10/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = module_2.generate_file(var_0, var_1, var_4, var_7)
    var_9 = '/fake/project/binary_file.png'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = {}
    var_5 = 'test'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'rendered content'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_12 = '/fake/project/template.txt'
    var_13 = 'w'
    var_14 = 'utf-8'
    var_15 = None

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_10 = bool(True)
    assert var_10 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{}}'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = ''
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7)
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_cookiecutter_new_lines_predicate. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_ensure_predicate_at_line_59_evaluates_to_true. Retrieved 17/18 statements.


import cookiecutter.utils as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_jinja2_env_vars'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.create_env_with_context(var_4)
    var_6 = 'test_repo'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = '{{cookiecutter.project_name}}'
    var_11 = var_9 / var_10
    var_12 = True
    var_13 = 'test_output'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_1.Path(*var_14, **var_15)
    var_17 = var_16.mkdir()
    var_18 = True
    var_19 = False
    var_20 = True
    var_21 = bool(var_20 and (not var_19 == var_18))
    assert var_21 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #55
#--------------------------




import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

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
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

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
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

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
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'empty_filename/'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'custom_newline.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/19 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 9/17 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 8/15 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 8/15 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/19 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'basic-template'
    var_7 = 'README.md'
    var_8 = 'test_project'

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
    var_7 = True

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
    var_9 = 'template-with-binaries'
    var_10 = 'data.bin'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'template-with-undefined'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_generate_context_default_context_none. Retrieved 3/4 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.generate_context(default_context=var_0)
    var_2 = 'cookiecutter'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/23 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 9/20 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 10/24 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 12/30 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/21 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = 'Modified content'
    var_9 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'print("Pre-hook executed")'
    var_11 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.md'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = '{{cookiecutter.project_name}}'
    var_9 = 'README.md'
    var_10 = 'This is a {{cookiecutter.project_name}} project.'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.undefined_var}}!'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_generate_context_with_default_context. Retrieved 8/12 statements.
# Partially parsed test_generate_context_with_extra_context. Retrieved 8/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test': {'key': 'value'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'new_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = {var_1: var_2}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'new_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = {var_1: var_2}

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = 'extra'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false_when_keep_project_on_failure_is_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #62
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_default_context_is_applied. Retrieved 9/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'cookiecutter.json'
    var_5 = 'key'
    var_6 = 'old_value'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_4, var_2, var_3)
    var_9 = var_8['cookiecutter']['key']
    assert var_9 == 'value'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false_when_output_directory_not_created_and_keep_project_on_failure_is_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #65
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #66
#--------------------------




def test_case_0():
    var_0 = bool(not (True and (not False)))
    assert var_0 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_1 = 'version'
    var_2 = '2.0.0'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'test_cookiecutter': {'name': 'test', 'version': '2.0.0'}})
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
    var_8 = bool(var_7 == {'test_cookiecutter': {'name': 'default', 'version': '2.0.0'}})
    assert var_8 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'non-existent.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'repo_dir'
    var_1 = 'hook_name'
    var_2 = 'project_dir'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_5, var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_render_and_create_dir_with_new_dir. Retrieved 7/8 statements.
# Partially parsed test_render_and_create_dir_with_template_rendering. Retrieved 9/10 statements.


import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_0, var_1, var_4, var_5)

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'existing_dir'
    var_7 = {}
    var_8 = var_3.parent
    var_9 = module_1.Environment()
    var_10 = module_2.render_and_create_dir(var_6, var_7, var_8, var_9)

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = {}
    var_7 = var_3.parent
    var_8 = module_1.Environment()
    var_9 = module_2.render_and_create_dir(var_0, var_6, var_7, var_8, var_4)
    var_10 = bool(var_9 == (var_3, False))
    assert var_10 is True

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'test_output'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'new_dir'
    var_5 = {}
    var_6 = module_1.Environment()
    var_7 = module_2.render_and_create_dir(var_4, var_5, var_3, var_6)
    var_8 = bool(var_7 == (var_3 / 'new_dir', True))
    assert var_8 is True
    var_9 = var_3 / var_4

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'test_output'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = '{{ project_name }}'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = module_1.Environment()
    var_9 = module_2.render_and_create_dir(var_4, var_7, var_3, var_8)
    var_10 = bool(var_9 == (var_3 / 'test_project', True))
    assert var_10 is True
    var_11 = var_3 / var_6



# Parsed testcases at query #4
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
    var_6 = {var_5: var_2}
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = bool(var_4 == {'existing': {'nested': 'value', 'new': 'value'}})
    assert var_9 is True

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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
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
    var_11 = bool(var_6 == {'config': {'key1': 'value1', 'key2': 'new_value2'}})
    assert var_11 is True

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



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'nested_key'
    var_4 = 'nested_value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_2, var_6)
    var_8 = bool(var_2 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'd'
    var_7 = 'e'
    var_8 = [var_6, var_7]
    var_9 = {var_0: var_8}
    var_10 = True
    var_11 = module_0.apply_overwrites_to_context(var_5, var_9, in_dictionary_variable=var_10)
    var_12 = var_5['key']
    var_13 = bool(var_5['key'] == ['d', 'e'])
    assert var_13 is True



# Parsed testcases at query #7
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
    var_5 = 'new_nested'
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.apply_overwrites_to_context(var_4, var_8, in_dictionary_variable=var_9)
    var_11 = bool(var_4 == {'existing': {'nested': 'value', 'new_nested': 'new_value'}})
    assert var_11 is True

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
    var_11 = 'd provided for choice variable list_var'

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
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = "['a', 'd'] provided for multi-choice variable multi_var"

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_val2'
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'dict_var': {'key1': 'val1', 'key2': 'new_val2'}})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
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
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'invalid provided for variable bool_var could not be converted to a boolean'

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



# Parsed testcases at query #8
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_0, var_1, var_4, var_5)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_render_and_create_dir_when_output_dir_exists. Retrieved 5/6 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_json_decoding_error_raises_context_decoding_exception. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = str(var_0)
    var_3 = 'JSON decoding error while loading'
    var_4 = bool('JSON decoding error while loading' in var_2)
    assert var_4 is True



# Parsed testcases at query #11
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
    var_0 = 'non_existent.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = True
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #13
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = module_1.Environment()
    var_7 = False
    var_8 = [var_5, var_0]
    var_9 = {}
    var_10 = module_0.Path(*var_8, **var_9)
    var_11 = True
    var_12 = var_10.mkdir(parents=var_11, exist_ok=var_11)
    var_13 = var_10.exists()
    assert var_13 is True



# Parsed testcases at query #14
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
    var_6 = 'x'
    var_7 = 'y'
    var_8 = [var_6, var_7]
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_5, var_9)
    var_11 = bool(var_5 == {'list_var': ['x', 'y']})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multi'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'multi': ['a', 'c']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multi'
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
    var_11 = "d provided for multi-choice variable multi, but valid choices are ['a', 'b', 'c']"

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
    var_10 = "d provided for choice variable choice, but the choices are ['a', 'b', 'c']"

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
    var_7 = 'invalid provided for variable bool_var could not be converted to a boolean'

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



# Parsed testcases at query #15
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
    var_11 = "d provided for choice variable choices, but the choices are ['a', 'b', 'c']."

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_1, var_3]
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(var_5 == {'multichoice': ['a', 'c']})
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'multichoice'
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
    var_11 = "['a', 'd'] provided for multi-choice variable multichoice, but valid choices are ['a', 'b', 'c']"

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'val1'
    var_4 = 'val2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'new_val2'
    var_8 = {var_2: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_6, var_9)
    var_11 = bool(var_6 == {'dict_var': {'key1': 'val1', 'key2': 'new_val2'}})
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
    var_7 = 'invalid provided for variable bool_var could not be converted to a boolean.'

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



# Parsed testcases at query #16
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'cookiecutter': {'name': 'test', 'value': 123}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'default', 'value': 123}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'extra', 'value': 123}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 14/21 statements.
# Partially parsed test_generate_file_text_file. Retrieved 14/21 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 15/22 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 16/21 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 14/19 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 14/19 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'binary_file.bin'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = True
    var_12 = b'\x00\x01\x02\x03'
    assert var_12 == b'\x00\x01\x02\x03'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

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
    var_10 = False
    var_11 = True
    var_12 = 'Hello, {{ name }}!'
    assert var_12 == 'Hello, {{ name }}!'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

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
    var_11 = True
    var_12 = 'Existing content'
    var_13 = 'Existing content'
    assert var_13 == 'Existing content'
    var_14 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

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
    var_12 = False
    var_13 = True
    var_14 = 'Content'
    var_15 = module_2.generate_file(var_0, var_1, var_8, var_11, var_12)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

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
    var_10 = False
    var_11 = True
    var_12 = 'Line 1\r\nLine 2'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

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
    var_10 = False
    var_11 = True
    var_12 = 'Line 1\r\nLine 2'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_cookiecutter_new_lines_predicate. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_file_name_is_empty. Retrieved 5/8 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'directory/'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = var_3.from_string(var_1)



# Parsed testcases at query #20
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #21
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_bool'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = var_2['my_bool']
    assert var_6 is True



# Parsed testcases at query #22
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #23
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
    var_7 = 'test_dir'
    var_8 = [var_4, var_7]
    var_9 = {}
    var_10 = module_2.Path(*var_8, **var_9)
    var_11 = var_6[0]
    var_12 = bool(var_6[0] == var_10)
    assert var_12 is True
    var_13 = var_6[1]
    assert var_13 is True

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
    var_6 = 'test_dir'
    var_7 = [var_4, var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = True
    var_11 = var_9.mkdir(exist_ok=var_10)
    var_12 = module_2.render_and_create_dir(var_0, var_3, var_4, var_5)

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
    var_6 = 'test_dir'
    var_7 = [var_4, var_6]
    var_8 = {}
    var_9 = module_1.Path(*var_7, **var_8)
    var_10 = True
    var_11 = var_9.mkdir(exist_ok=var_10)
    var_12 = module_2.render_and_create_dir(var_0, var_3, var_4, var_5, var_10)
    var_13 = [var_4, var_6]
    var_14 = {}
    var_15 = module_1.Path(*var_13, **var_14)
    var_16 = var_12[0]
    var_17 = bool(var_12[0] == var_15)
    assert var_17 is True
    var_18 = var_12[1]
    assert var_18 is False



# Parsed testcases at query #24
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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 13/20 statements.
# Partially parsed test_generate_file_text. Retrieved 15/22 statements.
# Partially parsed test_generate_file_skip_existing. Retrieved 13/18 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 13/18 statements.
# Partially parsed test_generate_file_configured_newline. Retrieved 13/18 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = b'\x00\x01\x02'
    assert var_11 == b'\x00\x01\x02'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = 'test'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello {{ name }}!'
    assert var_13 == 'Hello test!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'existing'
    assert var_11 == 'existing'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'newline.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'line1\r\nline2'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = b'\r\n'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'newline.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'line1\nline2'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = '\n'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_text_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 12/14 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 12/14 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 12/14 statements.
# Partially parsed test_generate_file_configured_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'empty_filename'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'configured_newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = b'\r\n'



# Parsed testcases at query #27
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_template_syntax_error_raises_with_translated_false. Retrieved 10/13 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = '{% invalid syntax %}'
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7)



# Parsed testcases at query #29
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test_cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_2 is True



# Parsed testcases at query #30
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_0, var_1, var_4, var_5)



# Parsed testcases at query #31
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = None
    var_2 = module_0.generate_context(var_0, var_1, var_1)
    var_3 = bool(var_2 == {})
    assert var_3 is True



# Parsed testcases at query #32
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test_cookiecutter': {'key': 'value'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/invalid_json.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_cookiecutter.json'
    var_1 = 'key'
    var_2 = 'default_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'test_cookiecutter': {'key': 'default_value'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_cookiecutter.json'
    var_1 = 'key'
    var_2 = 'extra_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'test_cookiecutter': {'key': 'extra_value'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_cookiecutter.json'
    var_1 = 'invalid_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_cookiecutter.json'
    var_1 = 'key'
    var_2 = 'invalid_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_nested_cookiecutter.json'
    var_1 = 'nested'
    var_2 = 'key'
    var_3 = 'nested_value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(var_6 == {'test_nested_cookiecutter': {'nested': {'key': 'nested_value'}}})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_list_cookiecutter.json'
    var_1 = 'choices'
    var_2 = 'valid_choice'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'test_list_cookiecutter': {'choices': ['valid_choice', 'other_choice']}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_list_cookiecutter.json'
    var_1 = 'choices'
    var_2 = 'valid_choice'
    var_3 = 'other_choice'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, extra_context=var_5)
    var_7 = bool(var_6 == {'test_list_cookiecutter': {'choices': ['valid_choice', 'other_choice']}})
    assert var_7 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_boolean_cookiecutter.json'
    var_1 = 'bool_var'
    var_2 = 'yes'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'test_boolean_cookiecutter': {'bool_var': True}})
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------




import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/path/to/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #34
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'README.md'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = module_0.is_copy_only_path(var_3, var_6)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'README.md'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.py'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False
    var_9 = 'src/main.py'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is False

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



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_template_syntax_error_raises_exception. Retrieved 10/13 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'fake_template.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '/fake/template/dir'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = '{% if %}'
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = '\n'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/10 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 9/11 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 9/11 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 9/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 9/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = False
    var_8 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_7)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 13/19 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = True
    var_12 = 'existing content'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_text_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/20 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 12/14 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
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
    var_0 = '/tmp/project'
    var_1 = '{{""}}'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = b'\r\n'



# Parsed testcases at query #40
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_skip_if_file_exists_and_outfile_exists. Retrieved 11/16 statements.


import jinja2.environment as module_0

def test_case_0():
    var_0 = '/fake/project/dir'
    var_1 = 'fake_infile.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = True
    var_10 = 'w'



# Parsed testcases at query #42
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = None
    var_2 = module_0.generate_context(var_0, var_1, var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_cookiecutter_new_lines_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]



# Parsed testcases at query #44
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = 'output'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)



# Parsed testcases at query #45
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/context.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'default', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/context.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/context.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/context.json'
    var_1 = None
    var_2 = module_0.generate_context(var_0, var_1, var_1)
    var_3 = bool(var_2 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_3 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_generate_file_with_binary_file. Retrieved 13/24 statements.
# Partially parsed test_generate_file_with_text_file. Retrieved 17/31 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/19 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 13/17 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.dat'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'binary.dat'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_12 = '/fake/project/binary.dat'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'template.txt'
    var_11 = 'template content'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = '/fake/project/template.txt'
    var_14 = 'w'
    var_15 = 'utf-8'
    var_16 = '\n'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'existing.txt'
    var_11 = True
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9, var_11)
    var_13 = '/fake/project/existing.txt'
    var_14 = bool(not var_9.get_template.called)
    assert var_14 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = ''
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_12 = '/fake/project'
    var_13 = bool(not var_9.get_template.called)
    assert var_13 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_at_line_36_evaluates_to_false. Retrieved 16/24 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 36 evaluates to False.'
    var_1 = '/path/to/repo'
    var_2 = 'cookiecutter'
    var_3 = '_jinja2_env_vars'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/path/to/output'
    var_8 = False
    var_9 = False
    var_10 = False
    var_11 = True
    var_12 = '/path/to/template'
    var_13 = '/path/to/project'
    var_14 = True
    var_15 = module_0.generate_files(var_1, var_6, var_7, var_8, var_9, var_10, var_11)
    var_16 = bool(var_15 == var_13)
    assert var_16 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 9/10 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 10/11 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 10/11 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 10/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 10/11 statements.
# Partially parsed test_generate_files_with_copy_only. Retrieved 12/13 statements.
# Partially parsed test_generate_files_with_new_lines. Retrieved 11/12 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test basic file generation.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_output'
    var_8 = module_0.generate_files(var_1, var_6, var_7)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with overwrite.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_output'
    var_8 = True
    var_9 = module_0.generate_files(var_1, var_6, var_7, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with skip existing files.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_output'
    var_8 = True
    var_9 = module_0.generate_files(var_1, var_6, var_7, skip_if_file_exists=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation without hooks.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_output'
    var_8 = False
    var_9 = module_0.generate_files(var_1, var_6, var_7, accept_hooks=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with keep on failure.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_output'
    var_8 = True
    var_9 = module_0.generate_files(var_1, var_6, var_7, keep_project_on_failure=var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with copy only paths.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = '_copy_without_render'
    var_5 = 'test_project'
    var_6 = '*.bin'
    var_7 = [var_6]
    var_8 = {var_3: var_5, var_4: var_7}
    var_9 = {var_2: var_8}
    var_10 = 'test_output'
    var_11 = module_0.generate_files(var_1, var_9, var_10)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with new lines configuration.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = '_new_lines'
    var_5 = 'test_project'
    var_6 = '\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 'test_output'
    var_10 = module_0.generate_files(var_1, var_8, var_9)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test file generation with undefined variable.'
    var_1 = 'test_repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_output'
    var_8 = module_0.generate_files(var_1, var_6, var_7)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/10 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 9/11 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 9/11 statements.
# Partially parsed test_generate_files_no_hooks. Retrieved 9/11 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 9/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = module_0.generate_files(var_5, var_4, var_6)
    var_8 = 'test_project'
    var_9 = bool('test_project' in var_7)
    assert var_9 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, var_7)
    var_9 = 'test_project'
    var_10 = bool('test_project' in var_8)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, skip_if_file_exists=var_7)
    var_9 = 'test_project'
    var_10 = bool('test_project' in var_8)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = False
    var_8 = module_0.generate_files(var_5, var_4, var_6, accept_hooks=var_7)
    var_9 = 'test_project'
    var_10 = bool('test_project' in var_8)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_repo'
    var_6 = 'test_output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, keep_project_on_failure=var_7)
    var_9 = 'test_project'
    var_10 = bool('test_project' in var_8)
    assert var_10 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_os_walk_returns_true. Retrieved 3/6 statements.


def test_case_0():
    var_0 = '.'
    var_1 = '../templates'
    var_2 = [var_0, var_1]



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_delete_project_on_failure_predicate_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_generate_file_with_binary_file. Retrieved 12/17 statements.
# Partially parsed test_generate_file_with_text_file. Retrieved 13/18 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/20 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 11/14 statements.
# Partially parsed test_generate_file_with_custom_newline. Retrieved 15/20 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = b'\x89PNG'
    var_10 = module_2.generate_file(var_0, var_1, var_4, var_7)
    var_11 = 'binary.png'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = {}
    var_5 = 'test'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello {{ name }}!'
    assert var_11 == 'Hello test!'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = {}
    var_5 = 'test'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'template.txt'
    var_12 = 'existing'
    assert var_12 == 'existing'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{}}'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7)
    var_10 = ''

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = '_new_lines'
    var_5 = '\r\n'
    var_6 = {var_4: var_5}
    var_7 = 'test'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello {{ name }}!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_15 = b'\r\n'
    var_16 = bool(b'\r\n' in var_13)
    assert var_16 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #3
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
    var_6 = {var_5: var_2}
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = bool(var_4 == {'existing': {'nested': 'value', 'new': 'value'}})
    assert var_9 is True

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
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_5, var_8)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'c'
    var_8 = 3
    var_9 = 4
    var_10 = {var_2: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11)
    var_13 = bool(var_6 == {'dict_var': {'a': 1, 'b': 3, 'c': 4}})
    assert var_13 is True

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



# Parsed testcases at query #4
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = 'post_gen_project'
    var_2 = '/path/to/project'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0._run_hook_from_repo_dir(var_0, var_1, var_2, var_7, var_8)
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #5
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'variable'
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_render_and_create_dir_existing_dir_no_overwrite. Retrieved 7/9 statements.
# Partially parsed test_render_and_create_dir_existing_dir_with_overwrite. Retrieved 10/11 statements.
# Partially parsed test_render_and_create_dir_new_dir. Retrieved 9/10 statements.
# Partially parsed test_render_and_create_dir_with_context. Retrieved 11/12 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1
import pathlib as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = [var_2]
    var_6 = {}
    var_7 = module_2.Path(*var_5, **var_6)
    var_8 = False
    var_9 = (var_7, var_8)
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = True
    var_2 = 'existing_dir'
    var_3 = {}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_2, var_3, var_4, var_5)

import jinja2.environment as module_0
import cookiecutter.generate as module_1
import pathlib as module_2

def test_case_0():
    var_0 = '/tmp/existing_dir'
    var_1 = True
    var_2 = 'existing_dir'
    var_3 = {}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_2, var_3, var_4, var_5, var_1)
    var_7 = [var_0]
    var_8 = {}
    var_9 = module_2.Path(*var_7, **var_8)
    var_10 = False
    var_11 = (var_9, var_10)
    var_12 = bool(var_6 == var_11)
    assert var_12 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1
import pathlib as module_2

def test_case_0():
    var_0 = 'new_dir'
    var_1 = {}
    var_2 = '/tmp'
    var_3 = module_0.Environment()
    var_4 = module_1.render_and_create_dir(var_0, var_1, var_2, var_3)
    var_5 = '/tmp/new_dir'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_2.Path(*var_6, **var_7)
    var_9 = True
    var_10 = (var_8, var_9)
    var_11 = bool(var_4 == var_10)
    assert var_11 is True

import jinja2.environment as module_0
import cookiecutter.generate as module_1
import pathlib as module_2

def test_case_0():
    var_0 = '{{ project_name }}'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = module_1.render_and_create_dir(var_0, var_3, var_4, var_5)
    var_7 = '/tmp/test_project'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_2.Path(*var_8, **var_9)
    var_11 = True
    var_12 = (var_10, var_11)
    var_13 = bool(var_6 == var_12)
    assert var_13 is True



# Parsed testcases at query #7
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'nested_key'
    var_4 = 'nested_value'
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_2, var_6)
    var_8 = bool(var_2 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 9/17 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 10/19 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 10/19 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 10/18 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 'Test basic file generation.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_data'
    var_7 = 'basic_template'
    var_8 = 'file.txt'

def test_case_0():
    var_0 = 'Test file generation with overwrite.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_data'
    var_7 = 'basic_template'
    var_8 = True
    var_9 = 'file.txt'

def test_case_0():
    var_0 = 'Test file generation with skip_if_file_exists.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_data'
    var_7 = 'basic_template'
    var_8 = True
    var_9 = 'file.txt'

def test_case_0():
    var_0 = 'Test file generation with hooks.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_data'
    var_7 = 'template_with_hooks'
    var_8 = True
    var_9 = 'file.txt'

def test_case_0():
    var_0 = 'Test file generation with copy_without_render.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = '_copy_without_render'
    var_4 = 'test_project'
    var_5 = '*.bin'
    var_6 = [var_5]
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = {var_1: var_7}
    var_9 = 'test_data'
    var_10 = 'template_with_binaries'
    var_11 = 'file.bin'



# Parsed testcases at query #9
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
    var_6 = 'new_value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = bool(var_4 == {'existing': {'nested': 'value', 'new': 'new_value'}})
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

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'config'
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
    var_11 = bool(var_6 == {'config': {'key1': 'value1', 'key2': 'new_value2'}})
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



# Parsed testcases at query #10
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'my_bool'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #11
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
    var_6 = 'x'
    var_7 = 'y'
    var_8 = [var_6, var_7]
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_5, var_9)
    var_11 = bool(var_5 == {'list_var': ['x', 'y']})
    assert var_11 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'choices'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = [var_2, var_1]
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
    var_6 = 'x'
    var_7 = 'y'
    var_8 = [var_6, var_7]
    var_9 = {var_0: var_8}
    var_10 = module_0.apply_overwrites_to_context(var_5, var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = "x, y provided for multi-choice variable choices, but valid choices are ['a', 'b', 'c']"

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
    var_6 = 'x'
    var_7 = {var_0: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_5, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = "x provided for choice variable choice, but the choices are ['a', 'b', 'c']"

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'dict_var'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'c'
    var_8 = 3
    var_9 = 4
    var_10 = {var_2: var_8, var_7: var_9}
    var_11 = {var_0: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_6, var_11)
    var_13 = bool(var_6 == {'dict_var': {'a': 1, 'b': 3, 'c': 4}})
    assert var_13 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'bool_var'
    var_1 = False
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
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'invalid provided for variable bool_var could not be converted to a boolean'

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



# Parsed testcases at query #12
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'cookiecutter.json'
    var_5 = module_0.generate_context(var_4, var_2, var_3)
    var_6 = 'cookiecutter'
    var_7 = bool('cookiecutter' in var_5)
    assert var_7 is True
    var_8 = var_5['cookiecutter']['key']
    assert var_8 == 'value'



# Parsed testcases at query #13
#--------------------------




import cookiecutter.generate as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = module_0.generate_files(var_5, var_4, var_6)
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Path(*var_8, **var_9)
    var_11 = var_10.exists()
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = [var_7, var_2]
    var_14 = {}
    var_15 = module_1.Path(*var_13, **var_14)
    var_16 = var_15.exists()
    var_17 = bool(var_16)
    assert var_17 is True

import cookiecutter.generate as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, var_7)
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_1.Path(*var_9, **var_10)
    var_12 = var_11.exists()
    var_13 = bool(var_12)
    assert var_13 is True
    var_14 = [var_8, var_2]
    var_15 = {}
    var_16 = module_1.Path(*var_14, **var_15)
    var_17 = var_16.exists()
    var_18 = bool(var_17)
    assert var_18 is True

import cookiecutter.generate as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, skip_if_file_exists=var_7)
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_1.Path(*var_9, **var_10)
    var_12 = var_11.exists()
    var_13 = bool(var_12)
    assert var_13 is True
    var_14 = [var_8, var_2]
    var_15 = {}
    var_16 = module_1.Path(*var_14, **var_15)
    var_17 = var_16.exists()
    var_18 = bool(var_17)
    assert var_18 is True

import cookiecutter.generate as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = False
    var_8 = module_0.generate_files(var_5, var_4, var_6, accept_hooks=var_7)
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_1.Path(*var_9, **var_10)
    var_12 = var_11.exists()
    var_13 = bool(var_12)
    assert var_13 is True
    var_14 = [var_8, var_2]
    var_15 = {}
    var_16 = module_1.Path(*var_14, **var_15)
    var_17 = var_16.exists()
    var_18 = bool(var_17)
    assert var_18 is True

import cookiecutter.generate as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = module_0.generate_files(var_5, var_4, var_6, keep_project_on_failure=var_7)
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_1.Path(*var_9, **var_10)
    var_12 = var_11.exists()
    var_13 = bool(var_12)
    assert var_13 is True
    var_14 = [var_8, var_2]
    var_15 = {}
    var_16 = module_1.Path(*var_14, **var_15)
    var_17 = var_16.exists()
    var_18 = bool(var_17)
    assert var_18 is True

import cookiecutter.generate as module_0
import pathlib as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.md'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'tests/test-template'
    var_9 = 'tests/output'
    var_10 = module_0.generate_files(var_8, var_7, var_9)
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_1.Path(*var_11, **var_12)
    var_14 = var_13.exists()
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = [var_10, var_3]
    var_17 = {}
    var_18 = module_1.Path(*var_16, **var_17)
    var_19 = var_18.exists()
    var_20 = bool(var_19)
    assert var_20 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_generate_context_raises_context_decoding_exception_on_invalid_json. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = str(var_0)
    var_3 = 'JSON decoding error while loading'
    var_4 = bool('JSON decoding error while loading' in var_2)
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = bool(not '' or '' == '')
    assert var_0 is True



# Parsed testcases at query #16
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'variable'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'invalid'
    var_4 = {var_0: var_3}
    var_5 = module_0.apply_overwrites_to_context(var_2, var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/26 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 9/24 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 9/24 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/29 statements.
# Partially parsed test_generate_files_hooks. Retrieved 14/38 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.md'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = '{{cookiecutter.project_name}}'
    var_9 = 'README.md'
    var_10 = 'Hello, {{cookiecutter.project_name}}!'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'test.txt'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = 'print("Pre-hook executed")'
    var_11 = 'post_gen_project.py'
    var_12 = 'print("Post-hook executed")'
    var_13 = True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_render_and_create_dir_creates_directory. Retrieved 7/9 statements.
# Partially parsed test_render_and_create_dir_overwrites_existing_directory. Retrieved 8/10 statements.
# Partially parsed test_render_and_create_dir_raises_exception_for_existing_directory. Retrieved 9/11 statements.


import pathlib as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/test_output'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = module_1.Environment()
    var_8 = '{{project_name}}'

import pathlib as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/test_output'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = module_1.Environment()
    var_8 = '{{project_name}}'
    var_9 = True

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/test_output'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = module_1.Environment()
    var_8 = ''
    var_9 = module_2.render_and_create_dir(var_8, var_2, var_6, var_7)

import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/test_output'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = module_1.Environment()
    var_8 = '{{project_name}}'
    var_9 = '{{project_name}}'
    var_10 = module_2.render_and_create_dir(var_9, var_2, var_6, var_7)



# Parsed testcases at query #19
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test-cookiecutter': {'name': 'test', 'version': '1.0'}})
    assert var_2 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 6/15 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 6/16 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 8/20 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/20 statements.
# Partially parsed test_generate_files_hooks. Retrieved 7/14 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'basic_template'
    var_5 = 'README.md'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'basic_template'
    var_5 = True

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'basic_template'
    var_5 = 'test'
    var_6 = True
    var_7 = 'new_file.txt'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'cookiecutter'
    var_2 = 'test_project'
    var_3 = '_copy_without_render'
    var_4 = '*.bin'
    var_5 = [var_4]
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = 'test_templates'
    var_9 = 'copy_template'
    var_10 = 'data.bin'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'hook_template'
    var_5 = True
    var_6 = 'hook_marker.txt'

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_templates'
    var_4 = 'undefined_template'



# Parsed testcases at query #21
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_0, var_1, var_4, var_5)



# Parsed testcases at query #22
#--------------------------




import pathlib as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = []
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1.Environment()
    var_6 = module_2.render_and_create_dir(var_0, var_1, var_4, var_5)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_generate_context_raises_context_decoding_exception_on_invalid_json. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = str(var_0)
    var_3 = 'JSON decoding error while loading'
    var_4 = bool('JSON decoding error while loading' in var_2)
    assert var_4 is True



# Parsed testcases at query #24
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'invalid.json'
    var_1 = module_0.generate_context(var_0)



# Parsed testcases at query #25
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'default', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
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
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_5 is True



# Parsed testcases at query #26
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'readme.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'docs/guide.md'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'script.py'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is False
    var_9 = 'src/main.py'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'readme.txt'
    var_4 = module_0.is_copy_only_path(var_3, var_2)
    assert var_4 is False

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'readme.txt'
    var_2 = module_0.is_copy_only_path(var_1, var_0)
    assert var_2 is False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_generate_file_with_binary_file. Retrieved 15/23 statements.
# Partially parsed test_generate_file_with_text_file. Retrieved 17/25 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 17/25 statements.
# Partially parsed test_generate_file_with_custom_newline. Retrieved 17/24 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = b'\x00\x01\x02\x03'
    assert var_11 == b'\x00\x01\x02\x03'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = 'binary.jpg'
    var_14 = module_3.rmtree(var_0)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = None
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello, {{ cookiecutter.name }}!'
    assert var_13 == 'Hello, test!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_15 = 'template.txt'
    var_16 = module_3.rmtree(var_0)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = None
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello, {{ cookiecutter.name }}!'
    var_14 = 'Existing content'
    assert var_14 == 'Existing content'
    var_15 = module_2.generate_file(var_0, var_1, var_8, var_11, var_12)
    var_16 = module_3.rmtree(var_0)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = '\r\n'
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello, {{ cookiecutter.name }}!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_15 = b'\r\n'
    var_16 = bool(var_3)
    assert var_16 is True
    var_17 = module_3.rmtree(var_0)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #29
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 4/9 statements.


def test_case_0():
    var_0 = True
    var_1 = 'existing_file.txt'
    var_2 = True
    var_3 = 'test'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 11/18 statements.
# Partially parsed test_generate_file_text_file. Retrieved 13/20 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 11/17 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 12/15 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 11/16 statements.
# Partially parsed test_generate_file_configured_newline. Retrieved 13/18 statements.


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
    var_8 = True
    var_9 = b'\x89PNG\r\n\x1a\n'
    assert var_9 == b'\x89PNG\r\n\x1a\n'
    var_10 = module_2.generate_file(var_0, var_1, var_4, var_7)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello {{ cookiecutter.name }}!'
    assert var_11 == 'Hello test!'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = 'existing content'
    assert var_9 == 'existing content'
    var_10 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{ cookiecutter.name }}/file.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = ''
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = 'Line 1\r\nLine 2\r\n'
    var_10 = module_2.generate_file(var_0, var_1, var_4, var_7)
    var_11 = b'\r\nLine 1\r\nLine 2\r\n'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'newline_file.txt'
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
    var_13 = b'Line 1\nLine 2\n'



# Parsed testcases at query #33
#--------------------------




import binaryornot.check as module_0

def test_case_0():
    var_0 = 'binary_file.png'
    var_1 = module_0.is_binary(var_0)
    assert var_1 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_skip_if_file_exists_predicate. Retrieved 4/9 statements.


def test_case_0():
    var_0 = True
    var_1 = 'existing_file.txt'
    var_2 = True
    var_3 = 'test'



# Parsed testcases at query #35
#--------------------------




import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = False
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = False
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'existing_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'empty_outfile.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = False
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = False
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'newline_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = False
    var_9 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_generate_context_with_invalid_default. Retrieved 7/13 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-project/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-project/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'override'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'override', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-project/cookiecutter.json'
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'cookiecutter': {'name': 'extra', 'version': '1.0.0'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-project/invalid.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'always'
    var_1 = 'tests/test-project/cookiecutter.json'
    var_2 = 'invalid'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_1, var_4)
    var_6 = 0
    var_7 = 'Invalid default received'
    var_8 = bool(var_5 == {'cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_8 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_generate_context_opens_file. Retrieved 5/7 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = bool(var_4 == {'test': {'key': 'value'}})
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_generate_context_opens_file. Retrieved 5/6 statements.


import codecs as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'w'
    var_2 = module_0.open(var_0, var_1)
    var_3 = '{"key": "value"}'
    var_4 = module_1.generate_context(var_0)
    var_5 = bool(var_4 == {'test': {'key': 'value'}})
    assert var_5 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 13/20 statements.
# Partially parsed test_generate_file_text. Retrieved 16/23 statements.
# Partially parsed test_generate_file_skip_existing. Retrieved 14/19 statements.
# Partially parsed test_generate_file_empty_name. Retrieved 11/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'binary.png'
    var_11 = True
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'template.txt'
    var_11 = True
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = 'w'
    var_14 = 'utf-8'
    var_15 = None

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'existing.txt'
    var_11 = True
    var_12 = True
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_12)
    var_14 = bool(not var_5)
    assert var_14 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{}}'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_11 = bool(not var_4)
    assert var_11 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_cookiecutter_new_lines_predicate. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_cookiecutter_new_lines_true. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_new_lines'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = var_4[var_0]
    var_6 = False



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_generate_file_binary. Retrieved 13/15 statements.
# Partially parsed test_generate_file_text. Retrieved 13/15 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
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
    var_12 = 'binary.png'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
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
    var_12 = 'template.txt'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/fake/template'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'empty_dir/'
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
    var_1 = 'template.txt'
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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_generate_file_binary_skipped_if_exists. Retrieved 12/17 statements.
# Partially parsed test_generate_file_text_skipped_if_exists. Retrieved 12/17 statements.
# Partially parsed test_generate_file_binary_copied. Retrieved 11/16 statements.
# Partially parsed test_generate_file_text_rendered. Retrieved 15/20 statements.
# Partially parsed test_generate_file_newline_detected. Retrieved 13/18 statements.
# Partially parsed test_generate_file_newline_configured. Retrieved 13/19 statements.


import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = True
    var_10 = 'binary.jpg'
    var_11 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = True
    var_10 = 'existing'
    assert var_10 == 'existing'
    var_11 = module_1.generate_file(var_0, var_1, var_6, var_7, var_8)

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.Environment()
    var_8 = True
    var_9 = b'binary content'
    assert var_9 == b'binary content'
    var_10 = module_1.generate_file(var_0, var_1, var_6, var_7)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = None
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello {{ cookiecutter.name }}!'
    assert var_13 == 'Hello test!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello\r\nWorld\r\n'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = b'\r\n'
    var_14 = bool(b'\r\n' in var_11)
    assert var_14 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'text.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello\r\nWorld\r\n'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = b'\r\n'
    var_14 = bool(b'\r\n' not in var_11)
    assert var_14 is True
    var_15 = b'\n'
    var_16 = bool(b'\n' in var_3)
    assert var_16 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 12/18 statements.
# Partially parsed test_generate_file_text_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 14/20 statements.
# Partially parsed test_generate_file_empty_outfile. Retrieved 12/14 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/14 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'binary_file.png'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
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
    var_11 = True
    var_12 = 'w'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = '{{""}}'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = 'text_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = b'\r\n'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 8/17 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 8/16 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 10/20 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 8/15 statements.
# Partially parsed test_generate_files_without_hooks. Retrieved 8/15 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 8/16 statements.
# Partially parsed test_generate_files_delete_on_failure. Retrieved 8/16 statements.


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

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'template-with-hooks'
    var_7 = False

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'failing-template'
    var_7 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test-data'
    var_6 = 'failing-template'
    var_7 = False



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_generate_file_binary_skip_if_exists. Retrieved 14/20 statements.
# Partially parsed test_generate_file_text_rendering. Retrieved 16/21 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 14/19 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 15/18 statements.
# Partially parsed test_generate_file_permissions. Retrieved 16/24 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.jpg'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
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
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = None
    var_6 = 'test'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = False
    var_13 = True
    var_14 = 'Hello {{ cookiecutter.name }}!'
    assert var_14 == 'Hello test!'
    var_15 = module_2.generate_file(var_0, var_1, var_8, var_11, var_12)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = True
    var_12 = 'Line 1\r\nLine 2\r\n'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_14 = b'\r\n'
    var_15 = bool(b'\r\n' in var_12)
    assert var_15 is True

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = '{{ cookiecutter.name }}.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = 'name'
    var_5 = None
    var_6 = ''
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = False
    var_13 = True
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11, var_12)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = True
    var_12 = 'test'
    var_13 = 420
    var_14 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_15 = 511



# Parsed testcases at query #50
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_generate_file_binary_file. Retrieved 10/14 statements.
# Partially parsed test_generate_file_text_file. Retrieved 12/16 statements.
# Partially parsed test_generate_file_skip_if_exists. Retrieved 12/18 statements.
# Partially parsed test_generate_file_empty_filename. Retrieved 10/11 statements.
# Partially parsed test_generate_file_newline_detection. Retrieved 12/16 statements.
# Partially parsed test_generate_file_custom_newline. Retrieved 12/16 statements.


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'binary.png'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

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
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = 'test'

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'existing.txt'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = True
    var_9 = True
    var_10 = 'w'
    var_11 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = ''
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'templates'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = None
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)

import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/fake/project'
    var_1 = 'template.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\r\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)
    var_12 = b'\r\n'



# Parsed testcases at query #52
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.apply_overwrites_to_context(var_3, var_2)
    var_5 = bool(var_3 == var_2)
    assert var_5 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_at_line_67_evaluates_to_false. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_2[var_0]
    var_4 = '_new_lines'
    var_5 = False



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_delete_project_on_failure_is_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 9/24 statements.
# Partially parsed test_generate_files_overwrite. Retrieved 10/25 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 10/25 statements.
# Partially parsed test_generate_files_copy_only. Retrieved 12/27 statements.
# Partially parsed test_generate_files_hooks. Retrieved 13/34 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    assert var_7 == 'Hello, test_project!'
    var_8 = 'test.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    assert var_7 == 'Hello, test_project!'
    var_8 = True
    var_9 = 'test.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = 'Hello, {{cookiecutter.project_name}}!'
    assert var_7 == 'Hello, test_project!'
    var_8 = True
    var_9 = 'test.txt'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.bin'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'template'
    var_9 = '{{cookiecutter.project_name}}'
    var_10 = b'\x00\x01\x02\x03'
    assert var_10 == b'\x00\x01\x02\x03'
    var_11 = 'test.bin'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = 'hooks'
    var_7 = 'print("Pre-hook executed")'
    var_8 = 'print("Post-hook executed")'
    var_9 = '{{cookiecutter.project_name}}'
    var_10 = 'Hello, {{cookiecutter.project_name}}!'
    assert var_10 == 'Hello, test_project!'
    var_11 = True
    var_12 = 'test.txt'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_predicate_at_line_59_evaluates_to_false. Retrieved 13/42 statements.


import cookiecutter.utils as module_0
import cookiecutter.find as module_1

def test_case_0():
    var_0 = '/path/to/repo'
    var_1 = {}
    var_2 = '/path/to/output'
    var_3 = False
    var_4 = False
    var_5 = True
    var_6 = True
    var_7 = '{{'
    var_8 = '}}'
    var_9 = module_0.create_env_with_context(var_1)
    var_10 = module_1.find_template(var_0, var_9)
    var_11 = 1
    var_12 = os.path.split(var_10)[var_11]



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 10/22 statements.
# Partially parsed test_generate_files_overwrite_existing. Retrieved 9/18 statements.
# Partially parsed test_generate_files_skip_existing_files. Retrieved 9/18 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 9/17 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 11/21 statements.
# Partially parsed test_generate_files_undefined_variable. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = 'README.md'
    var_8 = 'src'
    var_9 = 'main.py'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = True
    var_8 = 'README.md'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'basic_template'
    var_7 = 'existing content'
    assert var_7 == 'existing content'
    var_8 = True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'template_with_hooks'
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
    var_8 = 'test_data'
    var_9 = 'template_with_binary'
    var_10 = 'data.bin'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_data'
    var_6 = 'template_with_undefined'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_generate_context_with_invalid_default_context. Retrieved 7/11 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test': {'key': 'value'}})
    assert var_2 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = module_0.generate_context(var_0)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'new_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'new_value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'test': {'key': 'new_value'}})
    assert var_5 is True

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'c'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = "Invalid default received: ['c'] provided for multi-choice variable key, but valid choices are ['a', 'b']"

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'yes'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, extra_context=var_3)
    var_5 = bool(var_4 == {'test': {'key': True}})
    assert var_5 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_predicate_at_line_62_evaluates_to_false. Retrieved 20/32 statements.


import codecs as module_0
import cookiecutter.environment as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = 'test_dir'
    var_3 = [var_2]
    var_4 = {var_1: var_3}
    var_5 = {var_0: var_4}
    var_6 = True
    var_7 = 'render_dir'
    var_8 = 'test_file.txt'
    var_9 = 'w'
    var_10 = module_0.open(var_8, var_9)
    var_11 = True
    var_12 = 'context'
    var_13 = 'keep_trailing_newline'
    var_14 = {var_12: var_5, var_13: var_11}
    var_15 = module_1.StrictEnvironment(**var_14)
    var_16 = '.'
    var_17 = '../templates'
    var_18 = [var_16, var_17]
    var_19 = []
    var_20 = []
    var_21 = 'test_dir'
    var_22 = bool('test_dir' in var_19)
    assert var_22 is True
    var_23 = 'render_dir'
    var_24 = bool('render_dir' in var_20)
    assert var_24 is True
    var_25 = len(var_19)
    assert var_25 == 1
    var_26 = len(var_20)
    assert var_26 == 1



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_generate_context_opens_file. Retrieved 5/7 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = bool(var_4 == {'test': {'key': 'value'}})
    assert var_5 is True



# Parsed testcases at query #61
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #62
#--------------------------




import cookiecutter.generate as module_0
import collections as module_1

def test_case_0():
    var_0 = 'nonexistent.json'
    var_1 = None
    var_2 = module_0.generate_context(var_0, var_1, var_1)
    var_3 = 'nonexistent'
    var_4 = []
    var_5 = {}
    var_6 = module_1.OrderedDict(*var_4, **var_5)
    var_7 = {var_3: var_6}
    var_8 = bool(var_2 == var_7)
    assert var_8 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_generate_context_opens_file_with_utf8_encoding. Retrieved 3/5 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'utf-8'



# Parsed testcases at query #64
#--------------------------




import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = bool(var_1 == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}})
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
    var_0 = 'tests/test-cookiecutter.json'
    var_1 = 'invalid'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0, var_3)
    var_5 = bool(var_4 == {'test-cookiecutter': {'name': 'test', 'version': '1.0.0'}})
    assert var_5 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_generate_files_basic. Retrieved 11/12 statements.
# Partially parsed test_generate_files_skip_existing. Retrieved 11/13 statements.
# Partially parsed test_generate_files_with_hooks. Retrieved 11/12 statements.
# Partially parsed test_generate_files_keep_on_failure. Retrieved 11/12 statements.
# Partially parsed test_generate_files_copy_without_render. Retrieved 14/15 statements.


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = False
    var_9 = module_0.generate_files(var_5, var_4, var_6, var_7, var_8, var_8, var_8)
    assert var_9 == 'tests/output/test_project'
    var_10 = 'tests/output/test_project'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/output/test_project'
    var_6 = True
    var_7 = 'tests/test-template'
    var_8 = 'tests/output'
    var_9 = False
    var_10 = module_0.generate_files(var_7, var_4, var_8, var_9, var_6, var_9, var_9)
    assert var_10 == 'tests/output/test_project'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template-with-hooks'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = False
    var_9 = module_0.generate_files(var_5, var_4, var_6, var_7, var_8, var_7, var_8)
    assert var_9 == 'tests/output/test_project'
    var_10 = 'tests/output/test_project'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'tests/test-template'
    var_6 = 'tests/output'
    var_7 = True
    var_8 = False
    var_9 = module_0.generate_files(var_5, var_4, var_6, var_7, var_8, var_8, var_7)
    assert var_9 == 'tests/output/test_project'
    var_10 = 'tests/output/test_project'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'test_project'
    var_4 = '*.md'
    var_5 = [var_4]
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'tests/test-template'
    var_9 = 'tests/output'
    var_10 = True
    var_11 = False
    var_12 = module_0.generate_files(var_8, var_7, var_9, var_10, var_11, var_11, var_11)
    assert var_12 == 'tests/output/test_project'
    var_13 = 'tests/output/test_project'



