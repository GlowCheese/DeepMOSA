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
    var_1 = [var_0]
    var_2 = 'template'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = 'nested/template'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/to/dir/nested/template.json'



# Parsed testcases at query #2
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
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_without_suffix'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'invalid_template'
    var_2 = True
    var_3 = 'invalid json content'
    var_4 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'missing_cookiecutter_template'
    var_2 = True
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'cookiecutter'
    var_6 = bool('cookiecutter' in var_4)
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/11 statements.
# Partially parsed test_dump_raises_value_error_if_context_missing_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_handles_template_name_with_json_suffix. Retrieved 9/11 statements.


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
    var_6 = 'Context is required to contain a cookiecutter key'

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
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/14 statements.
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



# Parsed testcases at query #6
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test-template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_raises_valueerror_when_context_missing_cookiecutter. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'test-template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 9/10 statements.


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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/invalid/path/that/cannot/be/created'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/11 statements.


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



# Parsed testcases at query #10
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test-template'
    var_2 = 'some_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_dump_raises_valueerror_when_context_lacks_cookiecutter_key. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'other_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_predicate_at_line_5_evaluates_to_false.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 9/12 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_handles_template_name_with_json_suffix. Retrieved 9/12 statements.


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
    var_6 = 'Context is required to contain a cookiecutter key'

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
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'cookiecutter'
    var_6 = bool('cookiecutter' in var_4)
    assert var_6 is True



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_predicate_at_line_5_evaluates_to_false.




# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'cookiecutter'
    var_4 = bool('cookiecutter' in var_2)
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dump_raises_value_error_when_context_missing_cookiecutter. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'test-template'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_without_json_extension. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/13 statements.
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
    var_8 = 'test_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = 'Context is required to contain a cookiecutter key'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_with_valid_json. Retrieved 9/12 statements.
# Partially parsed test_load_without_json_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 7/11 statements.


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
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_without_suffix'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_missing_key'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #22
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
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

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
    var_8 = bool(var_7 == var_6)
    assert var_8 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_with_valid_file. Retrieved 9/12 statements.
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
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = [var_0]
    var_3 = f'{var_1}.json'
    var_4 = True
    var_5 = '{"cookiecutter": {}}'
    var_6 = 'utf-8'
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == {'cookiecutter': {}})
    assert var_8 is True



# Parsed testcases at query #26
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/replay'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'tests/replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'cookiecutter'
    var_6 = bool('cookiecutter' in var_4)
    assert var_6 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_replay_file_opened_successfully. Retrieved 3/6 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'test_template'
    var_2 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #30
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dump_creates_replay_dir_and_writes_json. Retrieved 9/14 statements.


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



# Parsed testcases at query #32
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
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

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
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #33
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dump_raises_value_error_when_context_missing_cookiecutter_key. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'test-template'
    var_3 = 'some_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dump_ensures_cookiecutter_key_in_context. Retrieved 7/9 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'cookiecutter'
    var_9 = bool('cookiecutter' in var_7)
    assert var_9 is True



# Parsed testcases at query #36
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_file.json'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template.json'
    var_3 = True
    var_4 = '{"cookiecutter": {"key": "value"}}'
    var_5 = 'utf-8'
    var_6 = 'test_template'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_with_valid_json_and_cookiecutter_key. Retrieved 9/15 statements.
# Partially parsed test_load_with_json_without_cookiecutter_key. Retrieved 7/14 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_replay_dir/test_template.json'
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'test_replay_dir/test_template.json'
    var_6 = module_0.load(var_0, var_1)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #41
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = '{"cookiecutter": {}}'
    var_5 = 'utf-8'



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
    var_1 = [var_0]
    var_2 = 'template'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = 'nested/template'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/to/dir/nested/template.json'



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'cookiecutter'
    var_6 = bool('cookiecutter' in var_4)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'path/to/replay_dir'
    var_1 = [var_0]
    var_2 = 'template_name'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/12 statements.
# Partially parsed test_dump_raises_value_error_if_no_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 9/12 statements.


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
    var_8 = 'test_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = 'Context is required to contain a cookiecutter key'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'test_template.json'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_dump_raises_valueerror_when_context_missing_cookiecutter_key. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'some_dir'
    var_1 = [var_0]
    var_2 = 'template'
    var_3 = 'some_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dump_creates_replay_file. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test_replays'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'test_replays/test_template.json'
    var_4 = [var_3]
    var_5 = True
    var_6 = '{"cookiecutter": {"key": "value"}}'
    var_7 = 'utf-8'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 10/14 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 10/14 statements.
# Partially parsed test_load_with_invalid_json_file. Retrieved 6/11 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 8/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = f'{var_1}.json'
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = f'{var_1}.json'
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = True
    var_3 = f'{var_1}.json'
    var_4 = 'invalid json'
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = f'{var_1}.json'
    var_7 = module_0.load(var_0, var_1)



# Parsed testcases at query #13
#--------------------------




import codecs as module_0

def test_case_0():
    var_0 = 'dummy_file.json'
    var_1 = 'utf-8'
    var_2 = module_0.open(var_0, encoding=var_1)
    var_3 = var_2.encoding
    assert var_3 == 'utf-8'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_file. Retrieved 9/12 statements.


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



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'cookiecutter'
    var_6 = bool('cookiecutter' in var_4)
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'some_dir'
    var_1 = [var_0]
    var_2 = 'some_template'
    var_3 = 'utf-8'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'cookiecutter'
    var_6 = bool('cookiecutter' in var_4)
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_dump_creates_replay_file. Retrieved 9/11 statements.
# Partially parsed test_dump_handles_json_suffix. Retrieved 8/10 statements.


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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_invalid_json_file. Retrieved 5/9 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = f'{var_1}.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = f'{var_1}.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = f'{var_1}.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = f'{var_1}.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.load(var_0, var_1)
    var_7 = 'Context is required to contain a cookiecutter key'



# Parsed testcases at query #21
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_replay_dir'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = 'cookiecutter'
    var_4 = bool('cookiecutter' in var_2)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = True
    var_5 = '{"cookiecutter": {}}'
    var_6 = 'utf-8'



# Parsed testcases at query #24
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
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_with_valid_json. Retrieved 9/12 statements.
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
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_with_valid_file. Retrieved 9/12 statements.
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
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/12 statements.
# Partially parsed test_dump_handles_template_name_with_json_suffix. Retrieved 9/10 statements.


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
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'utf-8'



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'cookiecutter'
    var_4 = bool('cookiecutter' in var_2)
    assert var_4 is True



