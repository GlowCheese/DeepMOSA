####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_pathlib_object. Retrieved 2/5 statements.


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

def test_case_0():
    var_0 = '/home/user/replays'
    var_1 = 'config'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '.'
    var_1 = ''
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == './.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_missing_cookiecutter_key. Retrieved 7/15 statements.
# Partially parsed test_dump_with_json_extension_already_present. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'test_replay'
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
    var_0 = 'test_replay'
    var_1 = 'my_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'wrong_key'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError not raised'
    var_6 = AssertionError(var_5)

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_predicate_false. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'some_key'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



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
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_error_dir'
    var_1 = 'invalid_config'
    var_2 = 'invalid_config.json'
    var_3 = True
    var_4 = 'not_cookiecutter'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'missing'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_success. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'test_replay.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_predicate_evaluates_to_false. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_dump_handles_json_extension_already_present. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'other'
    var_4 = 'name'
    var_5 = 'world'
    var_6 = {var_4: var_5}
    var_7 = 'data'
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'test_template.json'

def test_case_0():
    var_0 = 'replays'
    var_1 = 'test_template'
    var_2 = 'not_cookiecutter'
    var_3 = True
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'replays'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_exists. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template'



# Parsed testcases at query #9
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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = 'invalid'
    var_2 = 'invalid.json'
    var_3 = True
    var_4 = 'not_cookiecutter'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'ghost'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_success. Retrieved 10/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 8/17 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'test_template.json'
    var_3 = True
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_error_dir'
    var_1 = 'invalid_data.json'
    var_2 = 'invalid_data.json'
    var_3 = True
    var_4 = 'wrong_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'no_file.json'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_success. Retrieved 7/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
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
    var_3 = {}
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'non_existent'



# Parsed testcases at query #12
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'some_dir'
    var_1 = 'some_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #13
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/tmp/replay.json'
    var_6 = 'test_template'
    var_7 = module_0.load(var_5, var_6)



# Parsed testcases at query #14
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = '/tmp/replay'
    var_2 = '/tmp/replay/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_1, var_0)



# Parsed testcases at query #15
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_success. Retrieved 6/11 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.
# Partially parsed test_load_automatic_suffix_addition. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'

def test_case_0():
    var_0 = 'wrong_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'invalid.json'
    var_4 = 'invalid.json'

def test_case_0():
    var_0 = 'non_existent.json'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'template.json'
    var_4 = 'template'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_raises_error_when_cookiecutter_key_is_missing. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = 'some_other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'ValueError was not raised'
    var_6 = AssertionError(var_5)



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    pass

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'dummy_dir'
    var_1 = 'dummy_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_success. Retrieved 7/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_load_with_existing_extension. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = 'template'

def test_case_0():
    var_0 = 'wrong_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'invalid.json'
    var_4 = 'invalid'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = 'missing_file'
    var_2 = module_0.load(var_0, var_1)

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'template.json'
    var_4 = 'template.json'



# Parsed testcases at query #20
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'fake_dir'
    var_1 = 'fake_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_success. Retrieved 7/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
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
    var_2 = 'wrong_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'non_existent_file'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_evaluates_predicate_true. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'test_replay.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'template_name'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_dump_already_has_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_creates_nested_directories. Retrieved 7/17 statements.


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
    var_0 = 'replays'
    var_1 = 'test_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'data'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'replays'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'test_template.json'

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 'test'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = 'test.json'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_evaluates_predicate_true_when_cookiecutter_key_exists. Retrieved 9/14 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_replay.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_0.dumps(var_5)
    var_7 = 'utf-8'
    var_8 = 'template_name'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_is_missing. Retrieved 6/21 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay.json'
    var_1 = 'not_cookiecutter'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'template'
    var_5 = module_0.load(var_1, var_4)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_fails_when_file_does_not_exist. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'test_dir_non_existent'
    var_1 = True
    var_2 = 'non_existent_template'
    var_3 = f'{var_2}.json'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dump_predicate_evaluates_to_false. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #28
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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = 'invalid'
    var_2 = 'invalid.json'
    var_3 = True
    var_4 = 'wrong_key'
    var_5 = 'data'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'missing'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_file_not_found_raises_error. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test_dir_non_existent'
    var_1 = 'test_template'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_success. Retrieved 7/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
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
    var_2 = 'wrong_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'non_existent'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_success. Retrieved 7/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'config'
    var_1 = 'config.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'invalid_config'
    var_1 = 'invalid_config.json'
    var_2 = 'not_cookiecutter'
    var_3 = 'wrong_key'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'non_existent'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dump_predicate_evaluates_to_false. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'some'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_in_context. Retrieved 12/21 statements.


import json as module_0

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = f'{var_1}.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = 'utf-8'
    var_10 = globals()
    var_11 = 'get_file_name'



# Parsed testcases at query #34
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
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_1)

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



# Parsed testcases at query #35
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_template'
    var_1 = '/tmp/replays'
    var_2 = '/tmp/replays/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = module_1.load(var_1, var_0)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_json_extension. Retrieved 3/6 statements.
# Partially parsed test_get_file_name_without_json_extension. Retrieved 4/7 statements.
# Partially parsed test_get_file_name_with_pathlib_object. Retrieved 3/8 statements.
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

def test_case_0():
    var_0 = '/home/user/replay'
    var_1 = 'config'
    var_2 = 'config.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '.'
    var_1 = ''
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = '.json'



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_success. Retrieved 9/20 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 7/19 statements.
# Partially parsed test_load_file_not_found. Retrieved 2/7 statements.


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

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = True
    var_2 = 'invalid_template'
    var_3 = 'invalid_template.json'
    var_4 = 'wrong_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'no_file'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dump_predicate_false. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'some'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'



# Parsed testcases at query #5
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_directory'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/non_existent_path_12345'
    var_1 = 'test_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #6
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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = 'invalid_config'
    var_2 = 'invalid_config.json'
    var_3 = True
    var_4 = 'other_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'no_file'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_success. Retrieved 14/33 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'temp_test_dir'
    var_1 = True
    var_2 = 'test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = globals()
    var_9 = 'get_file_name'
    var_10 = 'test_template'
    var_11 = module_0.load(var_0, var_10)
    var_12 = 'get_file_name'
    var_13 = globals()[var_12]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dump_predicate_evaluates_to_false. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'some_key'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #9
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
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir_error'
    var_1 = 'invalid'
    var_2 = 'invalid.json'
    var_3 = True
    var_4 = 'wrong_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'missing'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_dump_handles_json_extension_already_present. Retrieved 6/14 statements.


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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_success. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = True
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #12
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
    var_7 = 123
    var_8 = {var_2: var_6, var_3: var_7}

def test_case_0():
    var_0 = 'invalid_template'
    var_1 = f'{var_0}.json'
    var_2 = 'not_cookiecutter'
    var_3 = True
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'non_existent_file'

def test_case_0():
    var_0 = 'existing_ext.json'
    var_1 = 'existing_ext.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}



# Parsed testcases at query #13
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = '/tmp/replay'
    var_2 = '/tmp/replay/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'my_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_1, var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_evaluates_predicate_true_with_valid_file. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = True
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'test'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_with_json_extension. Retrieved 6/14 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_dump_creates_directory_tree. Retrieved 8/19 statements.


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
    var_1 = 'already_has_extension.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'already_has_extension.json'

def test_case_0():
    var_0 = 'replays_error'
    var_1 = 'error_test'
    var_2 = 'not_cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'level1'
    var_1 = 'level2'
    var_2 = 'level3'
    var_3 = 'tree_test'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = 'tree_test.json'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_success. Retrieved 7/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 5/11 statements.
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
    var_2 = 'wrong_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'non_existent_file'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_missing_cookiecutter_key. Retrieved 5/14 statements.
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
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dump_success_path_exists_and_writes_json. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'my-template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_key_exists. Retrieved 11/19 statements.


import cookiecutter.replay as module_0

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
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #20
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
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'

def test_case_0():
    var_0 = 'non_existent'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_exists_in_context. Retrieved 15/23 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_replay.json'
    var_1 = 'cookiecutter'
    var_2 = 'other_key'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = 123
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = module_0.dumps(var_7)
    var_9 = 'utf-8'
    var_10 = globals()
    var_11 = 'get_file_name'
    var_12 = '.'
    var_13 = 'test_template'
    var_14 = module_1.load(var_12, var_13)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dump_predicate_false. Retrieved 11/17 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'some_key'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = '/tmp/replay/file.json'
    var_9 = 'w'
    var_10 = 'utf-8'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_raises_value_error_on_missing_cookiecutter_key. Retrieved 5/12 statements.
# Partially parsed test_dump_handles_json_extension_already_present. Retrieved 6/14 statements.


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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dump_writes_file_successfully. Retrieved 11/18 statements.


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



# Parsed testcases at query #25
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'fake_dir'
    var_1 = 'fake_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dump_writes_to_file_when_cookiecutter_key_exists. Retrieved 11/17 statements.


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



# Parsed testcases at query #27
#--------------------------




def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'some'
    var_4 = 'data'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'some'
    var_4 = 'data'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'not_cookiecutter'
    var_3 = True
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #28
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = '/tmp/replay/my_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'test_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_exists. Retrieved 10/16 statements.


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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dump_success. Retrieved 10/20 statements.
# Partially parsed test_dump_missing_cookiecutter_key. Retrieved 5/13 statements.
# Partially parsed test_dump_with_json_extension_already_present. Retrieved 6/14 statements.
# Partially parsed test_dump_handles_os_error_on_path_creation. Retrieved 6/16 statements.


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

def test_case_0():
    var_0 = 'blocked'
    var_1 = 'subdir'
    var_2 = 'test'
    var_3 = 'cookiecutter'
    var_4 = {}
    var_5 = {var_3: var_4}



# Parsed testcases at query #31
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
    var_2 = 'no_cookiecutter'
    var_3 = True
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'replays'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'my_template.json'



# Parsed testcases at query #32
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'some_dir'
    var_1 = 'some_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_dump_predicate_false_when_cookiecutter_in_context. Retrieved 11/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'some_key'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_0}/my_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = 'w'
    var_10 = 'utf-8'



# Parsed testcases at query #34
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'fake_dir'
    var_1 = 'fake_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_success_when_cookiecutter_in_context. Retrieved 12/21 statements.


def test_case_0():
    var_0 = 'replays'
    var_1 = 'test_template'
    var_2 = f'{var_1}.json'
    var_3 = 'cookiecutter'
    var_4 = 'other_key'
    var_5 = 'project_name'
    var_6 = 'my_project'
    var_7 = {var_5: var_6}
    var_8 = 'value'
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = globals()
    var_11 = 'get_file_name'



# Parsed testcases at query #36
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



