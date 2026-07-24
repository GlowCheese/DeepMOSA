####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_path_object. Retrieved 2/4 statements.


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

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = 'template.JSON'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/to/dir/template.JSON'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/13 statements.


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
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 6/12 statements.
# Partially parsed test_dump_raises_value_error_if_context_missing_cookiecutter_key. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_with_valid_file. Retrieved 3/4 statements.
# Partially parsed test_load_with_json_suffix. Retrieved 3/4 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/data'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/data'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/data'
    var_1 = 'valid_template.json'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #5
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'dummy_replay.json'
    var_1 = '{"cookiecutter": {}}'
    var_2 = 'utf-8'
    var_3 = 'dummy'



# Parsed testcases at query #7
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_with_valid_file. Retrieved 9/12 statements.
# Partially parsed test_load_without_json_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #9
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'test-template'
    var_2 = 'some'
    var_3 = 'context'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 6/8 statements.


import cookiecutter.replay as module_0
import codecs as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'utf-8'
    var_4 = module_1.open(var_2, encoding=var_3)
    var_5 = '{"cookiecutter": {}}'



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #12
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test-template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 9/13 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_handles_template_name_with_json_suffix. Retrieved 9/13 statements.


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
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_with_valid_json. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'valid_template'
    var_2 = True
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'valid_template.json'
    var_2 = True
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'invalid_template'
    var_2 = True
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/12 statements.


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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #16
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test-template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = {}



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 10/19 statements.
# Partially parsed test_load_with_invalid_json_file. Retrieved 8/17 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = f'{var_1}.json'
    var_9 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = True
    var_3 = f'{var_1}.json'
    var_4 = 'invalid_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = '{"cookiecutter": {}}'
    var_3 = 'utf-8'



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = {}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_with_valid_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.
# Partially parsed test_load_without_json_suffix. Retrieved 9/12 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent_file.json'



# Parsed testcases at query #25
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #26
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'test_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_with_valid_json_and_cookiecutter_key. Retrieved 9/12 statements.
# Partially parsed test_load_with_valid_json_without_cookiecutter_key. Retrieved 7/11 statements.
# Partially parsed test_load_with_json_suffix_in_template_name. Retrieved 9/12 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)



# Parsed testcases at query #28
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'Test that the file is opened with UTF-8 encoding.'
    var_1 = 'tests/replays'
    var_2 = 'test_template'
    var_3 = '{"cookiecutter": {}}'
    var_4 = 'utf-8'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json_file. Retrieved 9/14 statements.
# Partially parsed test_dump_raises_value_error_if_context_missing_cookiecutter_key. Retrieved 6/9 statements.


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
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #31
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_invalid_json_file. Retrieved 5/9 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_no_suffix'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_invalid'
    var_2 = True
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_open_file_with_utf8_encoding. Retrieved 8/10 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test'
    var_7 = module_0.load(var_0, var_6)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 4/6 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/data'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/data'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = str(var_0)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/data'
    var_1 = 'valid_template.json'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/data'
    var_1 = 'nonexistent_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_with_valid_json. Retrieved 3/4 statements.
# Partially parsed test_load_with_json_suffix. Retrieved 3/4 statements.
# Partially parsed test_load_without_json_suffix. Retrieved 3/4 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_replay_dir'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'invalid_replay_dir'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'template.json'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'replay_dir'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #36
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 9/13 statements.
# Partially parsed test_dump_handles_json_suffix_in_template_name. Retrieved 8/10 statements.


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
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

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



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'dummy_dir'
    var_1 = 'dummy_template'
    var_2 = f'{var_1}.json'
    var_3 = True
    var_4 = '{"cookiecutter": {}}'
    var_5 = 'utf-8'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_with_valid_json. Retrieved 3/4 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 4/6 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/data'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/data'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = str(var_0)
    assert var_3 == 'Context is required to contain a cookiecutter key'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/data'
    var_1 = 'template_with_suffix.json'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_invalid_json_file. Retrieved 5/9 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = True
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 9/11 statements.
# Partially parsed test_dump_uses_correct_file_name_with_json_suffix. Retrieved 8/10 statements.


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
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

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



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_invalid_json_file. Retrieved 5/9 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = True
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_load_with_valid_json. Retrieved 9/12 statements.
# Partially parsed test_load_without_json_extension. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_path_object. Retrieved 2/4 statements.


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

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = 'template'



# Parsed testcases at query #2
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
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_creates_replay_file. Retrieved 9/11 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_handles_json_suffix_in_template_name. Retrieved 8/10 statements.
# Partially parsed test_dump_writes_correct_context_to_file. Retrieved 8/10 statements.


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
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dump_creates_replay_dir_and_writes_context. Retrieved 9/14 statements.


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
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_with_valid_file. Retrieved 3/4 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/replay'
    var_1 = 'test_template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/replay'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/replay'
    var_1 = 'missing_cookiecutter'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/12 statements.


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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_with_valid_file. Retrieved 9/12 statements.
# Partially parsed test_load_without_json_extension. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/11 statements.
# Partially parsed test_dump_raises_value_error_if_context_missing_cookiecutter_key. Retrieved 6/9 statements.


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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #9
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #10
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #11
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'test-template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_invalid_json_file. Retrieved 5/9 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = True
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 10/12 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_0, var_1)
    var_9 = 'utf-8'



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #15
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #16
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dump_creates_file_with_correct_content. Retrieved 8/10 statements.


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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 5/8 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = 'test'
    var_2 = '{"cookiecutter": {}}'
    var_3 = module_0.load(var_0, var_1)
    var_4 = 'utf-8'



# Parsed testcases at query #19
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
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)



# Parsed testcases at query #20
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_replay_dir'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_with_valid_json. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_suffix. Retrieved 10/13 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 8/12 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.get_file_name(var_0, var_1)
    var_7 = module_0.load(var_0, var_1)



# Parsed testcases at query #22
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_with_valid_json. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_with_valid_file. Retrieved 10/15 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 8/14 statements.
# Partially parsed test_load_with_json_suffix. Retrieved 10/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.get_file_name(var_0, var_1)
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = module_0.load(var_0, var_1)



# Parsed testcases at query #25
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #26
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_replay_dir'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #27
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'invalid_replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/13 statements.
# Partially parsed test_dump_raises_value_error_if_context_missing_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 9/13 statements.


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
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_with_valid_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_with_valid_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_invalid_file. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_dir'
    var_1 = 'valid_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'invalid_dir'
    var_1 = 'invalid_template'
    var_2 = True
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'missing_dir'
    var_1 = 'missing_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #31
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'Test that the file is opened with UTF-8 encoding.'
    var_1 = 'test_replay.json'
    var_2 = '{"cookiecutter": {}}'
    var_3 = 'utf-8'
    var_4 = 'test_template'



# Parsed testcases at query #34
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #35
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #36
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #37
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}



# Parsed testcases at query #39
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_path'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #40
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 9/14 statements.


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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'dummy_replay.json'
    var_1 = 'test_template'
    var_2 = '{"cookiecutter": {}}'
    var_3 = 'utf-8'



# Parsed testcases at query #43
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_load_with_valid_json_and_cookiecutter_key. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_without_cookiecutter_key. Retrieved 7/11 statements.
# Partially parsed test_load_with_json_filename_without_extension. Retrieved 9/12 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)



# Parsed testcases at query #45
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
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
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = module_1.load(var_0, var_1)

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = module_1.load(var_0, var_1)



