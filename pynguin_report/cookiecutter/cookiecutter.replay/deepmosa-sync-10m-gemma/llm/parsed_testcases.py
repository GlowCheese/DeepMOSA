####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 8/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'config'
    var_2 = True
    var_3 = 'config.json'
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
    var_1 = 'invalid'
    var_2 = True
    var_3 = 'invalid.json'
    var_4 = 'wrong_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'missing'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replays'
    var_1 = 'data.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/tmp/replays/data.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replays'
    var_1 = 'data'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/tmp/replays/data.json'

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/home/user'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'config.json'
    var_5 = module_1.get_file_name(var_3, var_4)
    assert var_5 == '/home/user/config.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'logs'
    var_1 = ''
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == 'logs/.json'



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    pass

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/invalid/path'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'restricted.json'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_missing_cookiecutter_key. Retrieved 4/11 statements.
# Partially parsed test_dump_handles_json_extension_already_present. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'test'
    var_1 = 'not_cookiecutter'
    var_2 = True
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'template.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'template.json'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dump_predicate_false. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'some_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 'replay'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension_already_present. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 7/15 statements.
# Partially parsed test_dump_creates_nested_directories. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other_key'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'value'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays_ext'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'

def test_case_0():
    var_0 = 'replays_error'
    var_1 = 'test'
    var_2 = 'wrong_key'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError was not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 'level3'
    var_3 = 'test'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'test.json'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_raises_error_when_cookiecutter_key_missing. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'some_other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #8
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/fake/dir/template.json'
    var_6 = '/fake/dir'
    var_7 = 'template'
    var_8 = module_0.load(var_6, var_7)
    var_9 = bool(var_8 == var_4)
    assert var_9 is True
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_8)
    assert var_11 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 8/17 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'test_template'
    var_3 = 'test_template.json'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_2)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = True
    var_2 = 'invalid_template'
    var_3 = 'invalid_template.json'
    var_4 = 'not_cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_2)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_directory'
    var_1 = 'no_file_exists'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_success. Retrieved 7/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
# Partially parsed test_load_with_existing_extension. Retrieved 4/9 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'invalid_template'
    var_1 = f'{var_0}.json'
    var_2 = 'not_cookiecutter'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'already_has_ext.json'
    var_1 = 'cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'non_existent_file'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_success. Retrieved 10/12 statements.
# Partially parsed test_load_with_path_object. Retrieved 10/13 statements.


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
    var_7 = '/tmp/replay'
    var_8 = 'config'
    var_9 = 'config.json'
    var_10 = module_1.load(var_7, var_8)
    var_11 = bool(var_10 == var_4)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'wrong_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = '/tmp/replay'
    var_6 = 'config.json'
    var_7 = module_1.load(var_5, var_6)
    var_8 = 'ValueError was not raised'
    var_9 = AssertionError(var_8)

import json as module_0
import pathlib as module_1
import cookiecutter.replay as module_2

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = '/tmp/replay'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Path(*var_6, **var_7)
    var_9 = 'config'
    var_10 = 'config.json'
    var_11 = module_2.load(var_8, var_9)
    var_12 = bool(var_11 == {'cookiecutter': {}})
    assert var_12 is True
    var_13 = 'utf-8'



# Parsed testcases at query #12
#--------------------------




import pathlib as module_0
import cookiecutter.replay as module_1
import cookiecutter.utils as module_2

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_template'
    var_5 = 'cookiecutter'
    var_6 = 'some_key'
    var_7 = 'some_value'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = module_1.dump(var_3, var_4, var_9)
    var_11 = module_2.rmtree(var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dump_predicate_at_line_11_evaluates_to_false. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dump_writes_to_file_when_predicate_is_true. Retrieved 13/24 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'some_key'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '/tmp/replay/my_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = 'w'
    var_10 = 'utf-8'
    var_11 = ''
    var_12 = 0



# Parsed testcases at query #15
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/fake/dir'
    var_1 = 'test_template'
    var_2 = f'{var_0}/{var_1}.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'cookiecutter'
    var_13 = bool('cookiecutter' in var_10)
    assert var_13 is True



# Parsed testcases at query #16
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
    var_0 = 'replays_ext'
    var_1 = 'already_has_extension.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'already_has_extension.json'

def test_case_0():
    var_0 = 'replays_error'
    var_1 = 'error_test'
    var_2 = 'not_cookiecutter'
    var_3 = 'wrong_key'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError was not raised'
    var_6 = AssertionError(var_5)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/this_path_is_likely_forbidden/invalid_dir'
    var_1 = 'test'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = 'Unable to create directory at'
    var_7 = 'OSError was not raised for invalid path'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/16 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/10 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
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
    var_0 = 'error_dir'
    var_1 = 'test'
    var_2 = 'not_cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'ValueError not raised'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 7/15 statements.


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
    var_1 = 'invalid'
    var_2 = 'invalid.json'
    var_3 = True
    var_4 = 'not_cookiecutter'
    var_5 = {var_4: var_3}
    var_6 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'no_file'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dump_predicate_at_line_11_is_false. Retrieved 11/17 statements.


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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dump_writes_to_file_successfully. Retrieved 13/25 statements.


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
    var_11 = ''
    var_12 = 0



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_key_exists. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = 'value'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 'test_template.json'
    var_9 = 'cookiecutter'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_successfully_reads_json. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'test_replay.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template_name'
    var_7 = 'cookiecutter'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_evaluates_true_when_cookiecutter_exists. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = f'{var_1}.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_exists. Retrieved 9/18 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'other_key'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = 'value'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'cookiecutter'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_dump_raises_os_error_on_invalid_path. Retrieved 8/18 statements.


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
    var_0 = 'error_dir'
    var_1 = 'error_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError was not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'blocked_path'
    var_1 = 'sub_dir'
    var_2 = 'test'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'Unable to create directory at'
    var_7 = 'OSError was not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_success. Retrieved 14/17 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_replay_dir'
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
    var_11 = 'test_project'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = module_1.load(var_3, var_6)
    var_15 = bool(var_14 == var_13)
    assert var_15 is True
    var_16 = var_3.rmdir()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_dump_handles_json_extension_correctly. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'data'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'



# Parsed testcases at query #28
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = '/fake/path'
    var_2 = '/fake/path/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_1, var_0)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_8)
    assert var_11 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_evaluates_predicate_true_when_cookiecutter_exists. Retrieved 13/17 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_template'
    var_5 = f'{var_4}.json'
    var_6 = var_3 / var_5
    var_7 = True
    var_8 = var_3.mkdir(exist_ok=var_7)
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = 'my_project'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = module_1.load(var_3, var_4)
    var_15 = 'cookiecutter'
    var_16 = bool('cookiecutter' in var_14)
    assert var_16 is True
    var_17 = var_14['cookiecutter']['project_name']
    assert var_17 == 'my_project'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_exists. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'other_key'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = 'value'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'cookiecutter'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_success. Retrieved 7/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'

def test_case_0():
    var_0 = 'invalid_template'
    var_1 = 'not_cookiecutter'
    var_2 = 'wrong_key'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'non_existent'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/18 statements.
# Partially parsed test_dump_missing_cookiecutter_key. Retrieved 5/9 statements.
# Partially parsed test_dump_already_has_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_creates_nested_directories. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'other'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = 123
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 'my_template'
    var_8 = 'replay'
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'not_cookiecutter'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'test'
    var_4 = 'replay'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'test.json'
    var_4 = 'replay'
    var_5 = 'test.json'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'sub/dir/file'
    var_4 = 'outer/inner'
    var_5 = 'sub/dir/file.json'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_success_with_cookiecutter_key. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_dump_with_json_extension_already_present. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'data'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'



# Parsed testcases at query #35
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

# Partially parsed test_get_file_name_with_json_extension. Retrieved 3/6 statements.
# Partially parsed test_get_file_name_without_json_extension. Retrieved 4/7 statements.
# Partially parsed test_get_file_name_with_pathlib_object. Retrieved 4/7 statements.
# Partially parsed test_get_file_name_with_empty_template. Retrieved 4/7 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'data.json'
    var_2 = module_0.get_file_name(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'data'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'data.json'

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'config.json'
    var_5 = module_1.get_file_name(var_3, var_4)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '.'
    var_1 = ''
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = '.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_success. Retrieved 13/17 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 11/18 statements.
# Partially parsed test_load_file_not_found. Retrieved 6/12 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'config'
    var_7 = 'config.json'
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
    var_6 = 'invalid'
    var_7 = 'invalid.json'
    var_8 = var_3 / var_7
    var_9 = 'wrong_key'
    var_10 = 'no_cookiecutter_here'
    var_11 = {var_9: var_10}
    var_12 = module_1.load(var_3, var_6)

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir_missing'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'non_existent'
    var_5 = module_1.load(var_3, var_4)
    var_6 = var_3.exists()
    var_7 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_dump_raises_os_error_on_invalid_path. Retrieved 9/23 statements.


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
    var_0 = 'replays_error'
    var_1 = 'test'
    var_2 = 'not_cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'ValueError was not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'blocked_path'
    var_1 = 'i am a file'
    var_2 = 'new_subdir'
    var_3 = 'test'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'Unable to create directory at'
    var_8 = 'OSError was not raised'
    var_9 = AssertionError(var_8)



# Parsed testcases at query #4
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replays'
    var_1 = 'test_template'
    var_2 = f'{var_0}/{var_1}.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'cookiecutter'
    var_13 = bool('cookiecutter' in var_10)
    assert var_13 is True



# Parsed testcases at query #5
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'some'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/tmp/replay'
    var_6 = 'my-template'
    var_7 = module_0.dump(var_5, var_6, var_4)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/16 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/10 statements.
# Partially parsed test_dump_raises_value_error_when_missing_cookiecutter_key. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays_json'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'

def test_case_0():
    var_0 = 'replays_error'
    var_1 = 'test'
    var_2 = 'not_cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'ValueError was not raised'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #7
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = 'test_template'
    var_2 = '/tmp/test_dir/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------




import pathlib as module_0
import json as module_1
import cookiecutter.replay as module_2

def test_case_0():
    var_0 = 'test_replay.json'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'other_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = {}
    var_8 = module_1.dumps(var_6, **var_7)
    var_9 = 'utf-8'
    var_10 = var_3.write_text(var_8, var_9)
    var_11 = 'template_name'
    var_12 = module_2.load(var_0, var_11)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_dump_raises_os_error_on_invalid_path. Retrieved 8/18 statements.


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
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'

def test_case_0():
    var_0 = 'replays_error'
    var_1 = 'test'
    var_2 = 'wrong_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'blocked_by_file'
    var_1 = 'subfolder'
    var_2 = 'test'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'Unable to create directory at'
    var_7 = 'OSError not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_exists. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'cookiecutter'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dump_predicate_evaluates_to_false. Retrieved 11/17 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my-template'
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




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_template'
    var_1 = '/tmp/replay'
    var_2 = '/tmp/replay/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_1, var_0)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'cookiecutter'
    var_13 = bool('cookiecutter' in var_10)
    assert var_13 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 8/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'test_template.json'
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
    var_1 = 'invalid_template'
    var_2 = 'invalid_template.json'
    var_3 = True
    var_4 = 'wrong_key'
    var_5 = 'no_cookiecutter_here'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'ghost_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_exists. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'cookiecutter'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_success. Retrieved 13/17 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 13/22 statements.
# Partially parsed test_load_file_not_found. Retrieved 6/10 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'config'
    var_7 = 'config.json'
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
    var_6 = 'invalid'
    var_7 = 'invalid.json'
    var_8 = var_3 / var_7
    var_9 = 'not_cookiecutter'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = module_1.load(var_3, var_6)
    var_13 = 'Should have raised ValueError'
    assert var_13 == 'Context is required to contain a cookiecutter key'
    var_14 = AssertionError(var_13)
    var_15 = bool(var_12)
    assert var_15 is True

import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'ghost'
    var_5 = module_1.load(var_3, var_4)
    var_6 = 'Should have raised FileNotFoundError'
    var_7 = AssertionError(var_6)
    var_8 = bool(var_5)
    assert var_8 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_when_cookiecutter_key_missing. Retrieved 7/15 statements.
# Partially parsed test_dump_raises_os_error_on_invalid_path. Retrieved 8/18 statements.


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
    var_0 = 'replays_error'
    var_1 = 'test'
    var_2 = 'not_cookiecutter'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'file_collision'
    var_1 = 'subdir'
    var_2 = 'test'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'Unable to create directory at'
    var_7 = 'OSError not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 8/17 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'config'
    var_3 = 'config.json'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_2)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = True
    var_2 = 'invalid_config'
    var_3 = 'invalid_config.json'
    var_4 = 'wrong_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_2)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_directory'
    var_1 = 'no_file'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dump_writes_to_file_successfully. Retrieved 11/18 statements.


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



# Parsed testcases at query #19
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
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError was not raised'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_missing_cookiecutter_key. Retrieved 5/15 statements.
# Partially parsed test_dump_with_json_extension_already_present. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'not_cookiecutter'
    var_3 = True
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dump_predicate_at_line_11_is_false. Retrieved 11/17 statements.


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



# Parsed testcases at query #22
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'invalid_dir'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_success. Retrieved 7/13 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.


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
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'non_existent'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_success. Retrieved 14/19 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_replay_dir'
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
    var_11 = 'test_project'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = module_1.load(var_3, var_6)
    var_15 = bool(var_14 == var_13)
    assert var_15 is True
    var_16 = var_3.rmdir()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_dump_executes_file_writing_when_context_is_valid. Retrieved 10/22 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_template'
    var_5 = 'cookiecutter'
    var_6 = 'some_key'
    var_7 = 'some_value'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = f'{var_3}/{var_4}.json'
    var_11 = module_1.dump(var_3, var_4, var_9)



# Parsed testcases at query #26
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_directory'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'fake_dir'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'invalid_path'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_success. Retrieved 7/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.
# Partially parsed test_load_with_existing_json_extension. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

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
    var_0 = 'already_has_extension.json'
    var_1 = 'already_has_extension.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_invalid_json_raises_error. Retrieved 6/15 statements.
# Partially parsed test_load_missing_cookiecutter_key_raises_value_error. Retrieved 8/16 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'config'
    var_2 = True
    var_3 = 'config.json'
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
    var_0 = 'test_error_dir'
    var_1 = 'bad_data'
    var_2 = True
    var_3 = 'bad_data.json'
    var_4 = 'not a json'
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_missing_key_dir'
    var_1 = 'no_key'
    var_2 = True
    var_3 = 'no_key.json'
    var_4 = 'wrong_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'ghost'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_success_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_dump_raises_os_error_on_invalid_path. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
    var_5 = 'test_user'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'replays_alt'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'replays_error'
    var_1 = 'test_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'a_file.txt'
    var_1 = 'content'
    var_2 = 'test'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'Unable to create directory at'
    var_7 = 'OSError not raised for invalid path'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension_already_present. Retrieved 6/14 statements.
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
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'ValueError not raised'
    var_6 = AssertionError(var_5)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'test'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = 'Unable to create directory at'
    var_7 = 'OSError not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #31
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
    var_8 = '/tmp/dir'
    var_9 = 'template_name'
    var_10 = module_1.load(var_8, var_9)
    var_11 = 'cookiecutter'
    var_12 = bool('cookiecutter' in var_10)
    assert var_12 is True
    var_13 = var_10['cookiecutter']['project_name']
    assert var_13 == 'test_project'



# Parsed testcases at query #32
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
    var_7 = 'fake_dir'
    var_8 = 'test_template'
    var_9 = module_1.load(var_7, var_8)
    var_10 = bool(var_9 == var_4)
    assert var_10 is True
    var_11 = 'cookiecutter'
    var_12 = bool('cookiecutter' in var_9)
    assert var_12 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_dump_with_json_extension_already_present. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'my_template.json'

def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'data'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'



# Parsed testcases at query #34
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'fake_dir'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_exists_in_context. Retrieved 16/21 statements.


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
    var_10 = 'other_key'
    var_11 = 'project_name'
    var_12 = 'my_project'
    var_13 = {var_11: var_12}
    var_14 = 'value'
    var_15 = {var_9: var_13, var_10: var_14}
    var_16 = module_1.load(var_3, var_6)
    var_17 = 'cookiecutter'
    var_18 = bool('cookiecutter' in var_16)
    assert var_18 is True
    var_19 = var_16['cookiecutter']['project_name']
    assert var_19 == 'my_project'
    var_20 = var_3.rmdir()



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_load_file_exists_and_is_readable. Retrieved 14/17 statements.


import pathlib as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'test_template'
    var_7 = f'{var_6}.json'
    var_8 = var_3 / var_7
    var_9 = 'cookiecutter'
    var_10 = 'some_key'
    var_11 = 'some_value'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = module_1.load(var_3, var_6)
    var_15 = bool(var_14 == var_13)
    assert var_15 is True
    var_16 = var_3.rmdir()



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dump_writes_to_file_successfully. Retrieved 11/18 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my-template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test-project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_0}/{var_1}.json'
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = 'w'
    var_10 = 'utf-8'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
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
    var_1 = 'error_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'data'
    var_4 = {var_2: var_3}



# Parsed testcases at query #39
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
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



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_evaluates_true_when_cookiecutter_exists. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'test_replay.json'
    var_1 = 'cookiecutter'
    var_2 = 'other'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 'data'
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 'template_name'
    var_9 = 'cookiecutter'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_dump_writes_json_successfully. Retrieved 15/27 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'other_key'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = 'value'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = f'{var_0}/{var_1}.json'
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = 'w'
    var_12 = 'utf-8'
    var_13 = ''
    var_14 = 0



# Parsed testcases at query #42
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_template'
    var_1 = '/fake/path'
    var_2 = '/fake/path/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_1, var_0)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'cookiecutter'
    var_13 = bool('cookiecutter' in var_10)
    assert var_13 is True



