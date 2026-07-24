####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'data.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/tmp/replay/data.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'data'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/tmp/replay/data.json'

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'config'
    var_5 = module_1.get_file_name(var_3, var_4)
    assert var_5 == '/tmp/replay/config.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'logs'
    var_1 = ''
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == 'logs/.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_dump_raises_oserror_on_invalid_path. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
    var_5 = 'test_user'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays_ext'
    var_1 = 'already_has_extension.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'already_has_extension.json'

def test_case_0():
    var_0 = 'replays_error'
    var_1 = 'test'
    var_2 = 'not_cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'ValueError was not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'blocked_by_file'
    var_1 = 'sub_dir'
    var_2 = 'test'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'Unable to create directory at'
    var_7 = 'OSError was not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 8/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'config'
    var_2 = 'config.json'
    var_3 = True
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = 'invalid'
    var_2 = 'invalid.json'
    var_3 = True
    var_4 = 'wrong_key'
    var_5 = 'no_cookiecutter_here'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'missing'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #4
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'some'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/tmp/test_dir'
    var_6 = 'test_template'
    var_7 = module_0.dump(var_5, var_6, var_4)



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    pass

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'invalid_path'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_already_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_dump_raises_os_error_on_invalid_path. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays_ext'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'template.json'

def test_case_0():
    var_0 = 'replays_error'
    var_1 = 'test'
    var_2 = 'not_cookiecutter'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'ValueError not raised'
    var_6 = AssertionError(var_5)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'blocked_file'
    var_1 = 'sub_dir'
    var_2 = '/proc/invalid_permission_test_path'
    var_3 = 'test'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.dump(var_2, var_3, var_6)
    var_8 = 'Unable to create directory at'
    var_9 = 'OSError not raised for invalid path'
    var_10 = AssertionError(var_9)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_success. Retrieved 13/17 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 11/16 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'test_template'
    var_7 = 'test_template.json'
    var_8 = var_3 / var_7
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = 'my_project'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = module_1.load(var_3, var_6)
    var_15 = bool(var_14 == var_13)
    assert var_15 is True

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'invalid_template'
    var_7 = 'invalid_template.json'
    var_8 = var_3 / var_7
    var_9 = 'wrong_key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = module_1.load(var_3, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_directory'
    var_1 = 'missing_file'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------




import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/non_existent_replay_file_12345.json'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_template'
    var_5 = module_1.load(var_3, var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 8/16 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'config'
    var_2 = 'config.json'
    var_3 = True
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = 'invalid_config'
    var_2 = 'invalid_config.json'
    var_3 = True
    var_4 = 'wrong_key'
    var_5 = 'no_cookiecutter_here'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'missing'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_success. Retrieved 15/19 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 11/18 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'test_template'
    var_7 = 'test_template.json'
    var_8 = var_3 / var_7
    var_9 = 'cookiecutter'
    var_10 = 'other'
    var_11 = 'name'
    var_12 = 'world'
    var_13 = {var_11: var_12}
    var_14 = 123
    var_15 = {var_9: var_13, var_10: var_14}
    var_16 = module_1.load(var_3, var_6)
    var_17 = bool(var_16 == var_15)
    assert var_17 is True

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'invalid_template'
    var_7 = 'invalid_template.json'
    var_8 = var_3 / var_7
    var_9 = 'no_key'
    var_10 = 'here'
    var_11 = {var_9: var_10}
    var_12 = module_1.load(var_3, var_6)
    var_13 = bool(False)
    assert var_13 is True

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'non_existent'
    var_5 = module_1.load(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dump_predicate_is_true. Retrieved 11/18 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'some'
    var_4 = 'data'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_0}/{var_1}.json'
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = 'w'
    var_10 = 'utf-8'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_success. Retrieved 14/17 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'test_template'
    var_7 = f'{var_6}.json'
    var_8 = var_3 / var_7
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = 'my_project'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = module_1.load(var_3, var_6)
    var_15 = bool(var_14 == var_13)
    assert var_15 is True
    var_16 = var_3.rmdir()



# Parsed testcases at query #13
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'dummy_dir'
    var_8 = 'template_name'
    var_9 = module_1.load(var_7, var_8)
    var_10 = bool(var_9 == var_4)
    assert var_10 is True
    var_11 = 'cookiecutter'
    var_12 = bool('cookiecutter' in var_9)
    assert var_12 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension_already_present. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'replays_ext'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'replays_error'
    var_1 = 'test_template'
    var_2 = 'wrong_key'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError was not raised'
    var_6 = AssertionError(var_5)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/dev/null/invalid'
    var_1 = 'test'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = 'Unable to create directory at'
    var_7 = 'OSError was not raised for invalid path'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_dump_creates_directories_automatically. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'replays_ext'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'replays_fail'
    var_1 = 'test_template'
    var_2 = 'not_cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'ValueError not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'nested'
    var_1 = 'dir'
    var_2 = 'structure'
    var_3 = 'test'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_success. Retrieved 13/19 statements.
# Partially parsed test_load_missing_cookiecutter_key_raises_error. Retrieved 11/20 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'test_template'
    var_7 = 'test_template.json'
    var_8 = var_3 / var_7
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = 'my_project'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = module_1.load(var_3, var_6)
    var_15 = bool(var_14 == var_13)
    assert var_15 is True

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'bad_template'
    var_7 = 'bad_template.json'
    var_8 = var_3 / var_7
    var_9 = 'not_cookiecutter'
    var_10 = {}
    var_11 = {var_9: var_10}
    var_12 = module_1.load(var_3, var_6)
    var_13 = bool(False)
    assert var_13 is True

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'no_file'
    var_5 = module_1.load(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------




import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test_replay.json'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_template'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = str(var_3)
    var_11 = module_1.load(var_10, var_4)
    var_12 = bool(var_11 == var_9)
    assert var_12 is True
    var_13 = 'cookiecutter'
    var_14 = bool('cookiecutter' in var_11)
    assert var_14 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dump_writes_to_file_when_context_is_valid. Retrieved 11/18 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'some_key'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_0}/{var_1}.json'
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = 'w'
    var_10 = 'utf-8'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dump_writes_to_file_when_context_is_valid. Retrieved 12/21 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_0}/{var_1}.json'
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = 'w'
    var_10 = 'utf-8'
    var_11 = 0
    var_12 = 'test_project'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dump_writes_context_to_file_successfully. Retrieved 10/25 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'other_data'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 123
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = module_0.dump(var_1, var_0, var_7)
    var_9 = 'replay.json'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'

def test_case_0():
    var_0 = 'replays'
    var_1 = 'test'
    var_2 = 'not_cookiecutter'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = 'ValueError was not raised'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #22
#--------------------------




import pathlib as module_0
import json as module_1
import cookiecutter.replay as module_2

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_replay.json'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = 'test_template'
    var_10 = {}
    var_11 = module_1.dumps(var_4, **var_10)
    var_12 = 'utf-8'
    var_13 = var_8.write_text(var_11, var_12)
    var_14 = var_8.parent
    var_15 = str(var_14)
    var_16 = module_2.load(var_15, var_9)
    var_17 = bool(var_16 == var_4)
    assert var_17 is True
    var_18 = 'cookiecutter'
    var_19 = bool('cookiecutter' in var_16)
    assert var_19 is True
    var_20 = var_8.unlink()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_exists. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'test_replay.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 123
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 'template_name'
    var_9 = 'cookiecutter'



# Parsed testcases at query #24
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test_replay.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = 'test_template'
    var_9 = module_1.load(var_0, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True
    var_12 = var_9['cookiecutter']['project_name']
    assert var_12 == 'test_project'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_dump_success. Retrieved 14/22 statements.
# Partially parsed test_dump_with_already_json_extension. Retrieved 9/12 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = 0
    var_14 = bool(var_9 == var_8)
    assert var_14 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'
    var_6 = module_0.dump(var_0, var_1, var_4)
    var_7 = 'w'
    var_8 = 'utf-8'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_success. Retrieved 9/17 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_1]
    var_6 = {}
    var_7 = module_0.Path(*var_5, **var_6)
    var_8 = 'template'
    var_9 = module_1.load(var_7, var_8)
    var_10 = bool(var_9 == var_4)
    assert var_10 is True
    var_11 = var_7.unlink()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dump_writes_context_to_file. Retrieved 13/20 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = f'{var_0}/{var_1}.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'



# Parsed testcases at query #28
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replays'
    var_1 = 'test_template'
    var_2 = '/tmp/replays/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_8)
    assert var_11 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_success. Retrieved 13/17 statements.
# Partially parsed test_load_raises_value_error_on_missing_cookiecutter. Retrieved 11/18 statements.
# Partially parsed test_load_raises_file_not_found. Retrieved 4/7 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'test_template'
    var_7 = 'test_template.json'
    var_8 = var_3 / var_7
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = 'my_project'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = module_1.load(var_3, var_6)
    var_15 = bool(var_14 == var_13)
    assert var_15 is True

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'invalid_template'
    var_7 = 'invalid_template.json'
    var_8 = var_3 / var_7
    var_9 = 'not_cookiecutter'
    var_10 = {}
    var_11 = {var_9: var_10}
    var_12 = module_1.load(var_3, var_6)
    var_13 = bool(False)
    assert var_13 is True

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'ghost_template'
    var_5 = module_1.load(var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dump_writes_json_to_file. Retrieved 15/23 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = f'{var_0}/{var_1}.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = ''
    var_14 = 0



# Parsed testcases at query #31
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'other_data'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 123
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = 'dummy_dir'
    var_10 = 'dummy_template'
    var_11 = module_1.load(var_9, var_10)
    var_12 = bool(var_11 == var_6)
    assert var_12 is True
    var_13 = 'cookiecutter'
    var_14 = bool('cookiecutter' in var_11)
    assert var_14 is True



# Parsed testcases at query #32
#--------------------------




import pathlib as module_0
import json as module_1
import cookiecutter.replay as module_2

def test_case_0():
    var_0 = 'test_replay.json'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_template'
    var_5 = 'cookiecutter'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = {}
    var_11 = module_1.dumps(var_9, **var_10)
    var_12 = str(var_3)
    var_13 = module_2.load(var_12, var_4)
    var_14 = bool(var_13 == var_9)
    assert var_14 is True
    var_15 = 'cookiecutter'
    var_16 = bool('cookiecutter' in var_13)
    assert var_16 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = 'template.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/to/dir/template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = 'template'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/to/dir/template.json'

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'template'
    var_5 = module_1.get_file_name(var_3, var_4)
    assert var_5 == '/path/to/dir/template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = ''
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == 'data/.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'logs/replays'
    var_1 = 'session_1'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == 'logs/replays/session_1.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 8/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'config'
    var_2 = 'config.json'
    var_3 = True
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = 'config'
    var_2 = 'config.json'
    var_3 = True
    var_4 = 'not_cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'missing'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_dump_raises_os_error_on_invalid_path. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays_ext'
    var_1 = 'already_has_extension.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'already_has_extension.json'

def test_case_0():
    var_0 = 'errors'
    var_1 = 'fail_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'existing_file.txt'
    var_1 = 'i am a file'
    var_2 = 'test'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'Unable to create directory at'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 8/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'config'
    var_2 = 'config.json'
    var_3 = True
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = 'invalid_config'
    var_2 = 'invalid_config.json'
    var_3 = True
    var_4 = 'wrong_key'
    var_5 = 'no_cookiecutter_here'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'missing'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_success. Retrieved 7/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/6 statements.
# Partially parsed test_load_already_has_json_extension. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'invalid_template'
    var_1 = 'invalid_template.json'
    var_2 = 'wrong_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'non_existent'

def test_case_0():
    var_0 = 'existing_ext.json'
    var_1 = 'existing_ext.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}



# Parsed testcases at query #6
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_predicate_evaluates_to_false. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'some_key'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_success. Retrieved 9/14 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.
# Partially parsed test_load_with_already_suffixed_template. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = 123
    var_8 = {var_2: var_6, var_3: var_7}

def test_case_0():
    var_0 = 'invalid_template'
    var_1 = f'{var_0}.json'
    var_2 = 'not_cookiecutter'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'non_existent'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'complete_name.json'
    var_1 = 'complete_name.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dump_predicate_evaluates_to_false. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'not_cookiecutter'
    var_2 = True
    var_3 = {var_1: var_2}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_evaluates_true_when_cookiecutter_exists. Retrieved 10/18 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'test_template'
    var_2 = f'{var_1}.json'
    var_3 = 'cookiecutter'
    var_4 = 'other_key'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = 123
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = 'cookiecutter'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dump_predicate_evaluates_to_false. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    pass

def test_case_0():
    pass



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'not_cookiecutter'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'test_template'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_is_missing. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'other_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError was not raised'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'not_cookiecutter'
    var_2 = True
    var_3 = {var_1: var_2}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_evaluates_predicate_to_true_when_cookiecutter_exists. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'test_replay.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 123
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = ''
    var_9 = 'cookiecutter'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_is_missing. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'test_replay.json'
    var_1 = 'other_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = 'template_name'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_success. Retrieved 12/18 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 7/14 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'test_template'
    var_3 = 'test_template.json'
    var_4 = 'cookiecutter'
    var_5 = 'other'
    var_6 = 'project_name'
    var_7 = 'my_project'
    var_8 = {var_6: var_7}
    var_9 = 'value'
    var_10 = {var_4: var_8, var_5: var_9}
    var_11 = module_0.load(var_0, var_2)
    var_12 = bool(var_11 == var_10)
    assert var_12 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = True
    var_2 = 'invalid_template'
    var_3 = 'invalid_template.json'
    var_4 = 'not_cookiecutter'
    var_5 = {var_4: var_1}
    var_6 = module_0.load(var_0, var_2)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'missing_file'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'test_template'
    var_2 = f'{var_0}/{var_1}.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_8)
    assert var_11 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 8/17 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'template'
    var_2 = 'template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = True
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == var_7)
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = 'invalid_template'
    var_2 = 'invalid_template.json'
    var_3 = 'not_cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'ghost'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_success. Retrieved 9/14 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.
# Partially parsed test_load_with_already_json_extension. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = 'value'
    var_8 = {var_2: var_6, var_3: var_7}

def test_case_0():
    var_0 = 'invalid_template'
    var_1 = f'{var_0}.json'
    var_2 = 'wrong_key'
    var_3 = 'no_cookiecutter_here'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'non_existent'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'data.json'
    var_1 = 'data.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    pass

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'invalid_path'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_dump_creates_nested_directories. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays_ext'
    var_1 = 'already_has_extension.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'already_has_extension.json'

def test_case_0():
    var_0 = 'error_test'
    var_1 = 'template'
    var_2 = 'wrong_key'
    var_3 = 'no_cookiecutter_here'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 'level3'
    var_3 = 'test'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}



# Parsed testcases at query #25
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'dummy_dir'
    var_1 = 'dummy_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_dump_handles_os_error_on_invalid_path. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays_ext'
    var_1 = 'existing_extension.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'existing_extension.json'

def test_case_0():
    var_0 = 'replays_error'
    var_1 = 'test'
    var_2 = 'wrong_key'
    var_3 = 'data'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'blocked_path'
    var_1 = 'sub_dir'
    var_2 = 'test'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}



# Parsed testcases at query #27
#--------------------------




import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'invalid_path_12345.json'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'template'
    var_5 = module_1.load(var_3, var_4)



# Parsed testcases at query #28
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = '/tmp/replay'
    var_2 = '/tmp/replay/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'other_key'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = 'value'
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = module_0.load(var_1, var_0)
    var_11 = bool(var_10 == var_9)
    assert var_11 is True
    var_12 = 'cookiecutter'
    var_13 = bool('cookiecutter' in var_10)
    assert var_13 is True



