####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_path_object. Retrieved 3/7 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/test/dir'
    var_1 = 'test.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/test/dir/test.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/test/dir'
    var_1 = 'test'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/test/dir/test.json'

def test_case_0():
    var_0 = '/test/dir'
    var_1 = 'test'
    var_2 = '/test/dir/test.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_success. Retrieved 9/11 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 7/10 statements.
# Partially parsed test_load_json_extension. Retrieved 9/11 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = module_1.load(var_0, var_1)

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = module_1.load(var_0, var_1)

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = module_1.load(var_0, var_1)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/10 statements.
# Partially parsed test_dump_creates_json_file. Retrieved 9/13 statements.
# Partially parsed test_dump_handles_existing_json_suffix. Retrieved 9/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'test_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'test_template.json'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_contains_cookiecutter_key. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'



# Parsed testcases at query #5
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #6
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'some_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_returns_dict_with_cookiecutter_key. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #8
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_successfully_reads_json_file. Retrieved 4/8 statements.
# Partially parsed test_load_handles_template_name_without_json_extension. Retrieved 4/8 statements.
# Partially parsed test_load_handles_template_name_with_json_extension. Retrieved 3/7 statements.
# Partially parsed test_load_raises_value_error_when_missing_cookiecutter_key. Retrieved 4/9 statements.
# Partially parsed test_load_raises_file_not_found_error_when_file_does_not_exist. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = '{"cookiecutter": {"key": "value"}}'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = '{"cookiecutter": {"key": "value"}}'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = '{"cookiecutter": {"key": "value"}}'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = '{"key": "value"}'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'nonexistent_template'



# Parsed testcases at query #10
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template_missing_key.json'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)



# Parsed testcases at query #11
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #12
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_extension_in_name. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'valid_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = '/tmp/valid_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = '/tmp/template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'invalid_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)
    var_6 = '/tmp/invalid_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'nonexistent'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_with_invalid_replay_file. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'invalid_dir'
    var_1 = 'invalid_template'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_without_cookiecutter_key_raises_value_error. Retrieved 3/10 statements.


def test_case_0():
    var_0 = '{"some_key": "some_value"}'
    var_1 = 'fake_dir'
    var_2 = 'fake_template'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dump_raises_error_when_cookiecutter_not_in_context. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'not_cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_with_valid_file_and_context. Retrieved 8/10 statements.
# Partially parsed test_load_with_invalid_context. Retrieved 6/9 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template_invalid.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'non_existent_template.json'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_returns_dict_with_cookiecutter_key. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = '{"cookiecutter": {"key": "value"}}'
    var_3 = globals()
    var_4 = 'get_file_name'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_success. Retrieved 8/10 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 6/9 statements.
# Partially parsed test_load_with_json_extension. Retrieved 8/10 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'invalid_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)



# Parsed testcases at query #20
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template_name'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_valid_file. Retrieved 8/10 statements.
# Partially parsed test_load_valid_file_without_json_suffix. Retrieved 8/10 statements.
# Partially parsed test_load_invalid_file_missing_cookiecutter. Retrieved 6/9 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'valid_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'valid_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'invalid_template.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'nonexistent_template.json'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #22
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'some_dir'
    var_1 = 'some_template'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_valid_template. Retrieved 8/10 statements.
# Partially parsed test_load_template_without_cookiecutter. Retrieved 6/9 statements.
# Partially parsed test_load_template_with_json_suffix. Retrieved 8/10 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'valid_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'invalid_template.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_with_invalid_replay_file. Retrieved 2/5 statements.


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = 'invalid_template'



# Parsed testcases at query #25
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'not_cookiecutter'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #26
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #27
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dump_creates_file_with_valid_context. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #29
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'missing_cookiecutter_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #30
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'not_cookiecutter'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'template'
    var_4 = 'path/to/replay'
    var_5 = module_0.dump(var_4, var_3, var_2)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dump_writes_to_file. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_file. Retrieved 9/19 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_dump_creates_file_with_context. Retrieved 9/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #34
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/path/to/replay_dir'
    var_2 = 'template_name'
    var_3 = module_0.load(var_1, var_2)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 7/10 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 4/7 statements.
# Partially parsed test_dump_creates_correct_json_file. Retrieved 7/13 statements.
# Partially parsed test_dump_handles_existing_json_suffix. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template.json'

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template.json'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_load_should_return_dict_when_file_exists_and_contains_cookiecutter_key. Retrieved 11/14 statements.
# Partially parsed test_load_should_raise_value_error_when_file_exists_but_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_3: var_4}
    var_8 = {var_2: var_7}
    var_9 = f'{var_1}.json'
    var_10 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = f'{var_1}.json'
    var_6 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'nonexistent_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_successfully_reads_valid_json_file. Retrieved 6/20 statements.
# Partially parsed test_load_raises_error_when_cookiecutter_key_missing. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'valid_template'
    var_2 = 'test_dir/valid_template.json'
    var_3 = '{"cookiecutter": {"key": "value"}}'
    var_4 = '.json'
    var_5 = var_1 + var_4

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'invalid_template'
    var_2 = 'test_dir/invalid_template.json'
    var_3 = '{"key": "value"}'
    var_4 = '.json'
    var_5 = var_1 + var_4



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dump_writes_file. Retrieved 9/14 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = f'{var_1}.json'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_dump_successfully_writes_to_file. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'



# Parsed testcases at query #41
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'nonexistent.json'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'invalid.json'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_load_returns_context_with_cookiecutter_key. Retrieved 9/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '{"cookiecutter": {"key": "value"}}'
    var_1 = 'mock_file.json'
    var_2 = lambda file, encoding: var_0
    var_3 = lambda replay_dir, template_name: var_1
    var_4 = var_2
    var_5 = var_3
    var_6 = 'mock_replay_dir'
    var_7 = 'mock_template'
    var_8 = module_0.load(var_6, var_7)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_with_non_existent_file. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'non_existent.json'
    var_1 = 'template'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_load_contains_cookiecutter_key. Retrieved 9/15 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = '/test'
    var_7 = 'template'
    var_8 = module_1.load(var_1, var_7)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 10/14 statements.
# Partially parsed test_dump_writes_correct_json_content. Retrieved 11/16 statements.
# Partially parsed test_dump_uses_correct_file_path. Retrieved 13/17 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter.replay.make_sure_path_exists'
    var_1 = 'builtins.open'
    var_2 = '/test/dir'
    var_3 = 'template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.dump(var_2, var_3, var_8)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/test/dir'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter.replay.make_sure_path_exists'
    var_1 = 'builtins.open'
    var_2 = '/test/dir'
    var_3 = 'template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.dump(var_2, var_3, var_8)
    var_10 = '{\n  "cookiecutter": {\n    "key": "value"\n  }\n}'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter.replay.make_sure_path_exists'
    var_1 = 'builtins.open'
    var_2 = '/test/dir'
    var_3 = 'template'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.dump(var_2, var_3, var_8)
    var_10 = '/test/dir/template.json'
    var_11 = 'w'
    var_12 = 'utf-8'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_load_successfully_reads_json_file. Retrieved 3/7 statements.
# Partially parsed test_load_adds_json_extension_if_missing. Retrieved 5/9 statements.
# Partially parsed test_load_raises_value_error_when_no_cookiecutter_key. Retrieved 3/8 statements.
# Partially parsed test_load_handles_path_object_input. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = '{"cookiecutter": {"key": "value"}}'

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = '{"cookiecutter": {"key": "value"}}'
    var_3 = -5
    var_4 = var_0[:var_3]

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = '{"key": "value"}'

def test_case_0():
    var_0 = 'test_template'
    var_1 = f'{var_0}.json'
    var_2 = '{"cookiecutter": {"key": "value"}}'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_dump_handles_invalid_context. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_load_with_cookiecutter_key. Retrieved 6/12 statements.


import json as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = '/path/to/replay'
    var_4 = 'template.json'
    var_5 = module_0.dumps(var_2)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_dump_creates_file_with_cookiecutter_context. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'



# Parsed testcases at query #50
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_path_object_and_json_suffix. Retrieved 2/4 statements.
# Partially parsed test_get_file_name_with_path_object_and_no_suffix. Retrieved 2/4 statements.


def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template.json'

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/to/replay/template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/to/replay/template.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_success. Retrieved 8/10 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_load_with_json_extension. Retrieved 8/10 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'invalid_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_with_non_existent_file. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/12 statements.
# Partially parsed test_dump_writes_correct_json_file. Retrieved 9/17 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = f'{var_1}.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #5
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'some_dir'
    var_1 = 'some_template'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #6
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_with_valid_replay_file_opens_file_successfully. Retrieved 9/15 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = '/test/dir'
    var_7 = 'template'
    var_8 = module_1.load(var_1, var_7)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_with_valid_cookiecutter_key. Retrieved 6/15 statements.
# Partially parsed test_load_without_cookiecutter_key_raises_error. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template_name'

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'template_name'



# Parsed testcases at query #9
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'dummy_dir'
    var_4 = 'dummy_template'
    var_5 = module_0.load(var_3, var_4)



# Parsed testcases at query #10
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = 'dummy_path'
    var_4 = 'dummy_template'
    var_5 = module_0.load(var_3, var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_raises_error_when_cookiecutter_not_in_context. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 'other_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'template'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_valid_file. Retrieved 9/11 statements.
# Partially parsed test_load_invalid_file_missing_cookiecutter. Retrieved 7/10 statements.
# Partially parsed test_load_file_without_json_extension. Retrieved 9/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = f'{var_1}.json'
    var_6 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'
    var_8 = module_0.load(var_0, var_1)



# Parsed testcases at query #13
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/some/dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #14
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'test_template'
    var_4 = 'some/directory'
    var_5 = module_0.dump(var_4, var_3, var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_valid_file. Retrieved 9/12 statements.
# Partially parsed test_load_file_without_cookiecutter_key. Retrieved 7/11 statements.
# Partially parsed test_load_file_without_json_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_file_with_json_suffix. Retrieved 8/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'valid_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'invalid_template'
    var_2 = f'{var_1}.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = f'{var_1}.json'
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_is_missing. Retrieved 4/16 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.load(var_1, var_2)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_with_valid_cookiecutter_key. Retrieved 8/16 statements.
# Partially parsed test_load_without_cookiecutter_key_raises_error. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'valid_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = f'{var_0}.json'
    var_7 = var_1 / var_6

def test_case_0():
    var_0 = 'invalid_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = f'{var_0}.json'
    var_5 = var_1 / var_4



# Parsed testcases at query #18
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'some/path'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_contains_cookiecutter_key. Retrieved 4/11 statements.


def test_case_0():
    var_0 = '{"cookiecutter": {"key": "value"}}'
    var_1 = 'fake_path'
    var_2 = 'template.json'
    var_3 = 'template'



# Parsed testcases at query #20
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/6 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = 'test_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #22
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'not_cookiecutter'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'test_template'
    var_4 = '/path/to/replay_dir'
    var_5 = module_0.dump(var_4, var_3, var_2)



# Parsed testcases at query #23
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_successfully_reads_json_file. Retrieved 8/10 statements.
# Partially parsed test_load_raises_error_when_cookiecutter_key_missing. Retrieved 6/9 statements.
# Partially parsed test_load_appends_json_suffix_if_missing. Retrieved 8/10 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_context_contains_cookiecutter_key. Retrieved 3/9 statements.


def test_case_0():
    var_0 = '{"cookiecutter": {"key": "value"}}'
    var_1 = 'fake_dir'
    var_2 = 'fake_template'



# Parsed testcases at query #26
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'invalid_dir'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #28
#--------------------------






# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_without_cookiecutter_key_raises_value_error. Retrieved 3/11 statements.


def test_case_0():
    var_0 = '{"some_key": "some_value"}'
    var_1 = '/fake/path'
    var_2 = 'template'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_with_valid_replay_file_opens_file_successfully. Retrieved 9/15 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = '/test/dir'
    var_7 = 'template'
    var_8 = module_1.load(var_1, var_7)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 7/14 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'not_cookiecutter'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    var_4 = 'test_dir'
    var_5 = 'template'
    var_6 = module_1.load(var_1, var_5)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/14 statements.
# Partially parsed test_dump_writes_correct_content_to_file. Retrieved 10/19 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = f'{var_1}.json'
    var_9 = var_3 / var_8

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #33
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_function_returns_valid_context_with_cookiecutter_key. Retrieved 6/15 statements.
# Partially parsed test_load_function_raises_error_when_cookiecutter_key_is_missing. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template_name'

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'template_name'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_with_invalid_replay_file. Retrieved 3/11 statements.


def test_case_0():
    var_0 = '{"not_cookiecutter": "value"}'
    var_1 = '/fake/path'
    var_2 = 'template'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/10 statements.
# Partially parsed test_dump_writes_correct_json_file. Retrieved 9/14 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = f'{var_1}.json'



# Parsed testcases at query #37
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'some_dir'
    var_4 = 'some_template'
    var_5 = module_0.load(var_3, var_4)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_dump_creates_file_with_correct_content. Retrieved 9/13 statements.
# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/9 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = f'{var_1}.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_context_contains_cookiecutter_key. Retrieved 8/10 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_raises_error_when_cookiecutter_not_in_context. Retrieved 5/11 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.dumps(var_0)
    var_2 = 'fake_dir'
    var_3 = 'fake_template'
    var_4 = module_1.load(var_2, var_3)



# Parsed testcases at query #41
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'fake_dir'
    var_2 = 'fake_template'
    var_3 = module_0.load(var_1, var_2)



# Parsed testcases at query #42
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'some/dir'
    var_1 = 'template'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_success. Retrieved 9/11 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 7/10 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/test/dir'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = module_1.load(var_0, var_1)

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/test/dir'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = module_1.load(var_0, var_1)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/12 statements.
# Partially parsed test_dump_raises_value_error_if_no_cookiecutter_key. Retrieved 6/10 statements.
# Partially parsed test_dump_creates_json_file_with_correct_content. Retrieved 9/17 statements.
# Partially parsed test_dump_handles_existing_json_suffix_correctly. Retrieved 8/14 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = f'{var_1}.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_load_returns_context_with_cookiecutter_key. Retrieved 5/17 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = '{"cookiecutter": {"key": "value"}}'
    var_3 = 'test_dir/test_template.json'
    var_4 = module_0.load(var_0, var_1)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_load_successful_with_json_suffix. Retrieved 8/10 statements.
# Partially parsed test_load_successful_without_json_suffix. Retrieved 8/10 statements.
# Partially parsed test_load_raises_value_error_when_missing_cookiecutter_key. Retrieved 6/9 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'invalid_template.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/nonexistent'
    var_1 = 'missing_template.json'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #47
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_load_raises_value_error_when_context_does_not_contain_cookiecutter_key. Retrieved 5/16 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'fake_dir'
    var_1 = 'fake_template'
    var_2 = {}
    var_3 = 'Context is required to contain a cookiecutter key'
    var_4 = module_0.load(var_0, var_1)



