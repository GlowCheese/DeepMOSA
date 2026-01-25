####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'choice_var'
    var_2 = 'multi_choice_var'
    var_3 = 'bool_var'
    var_4 = 'dict_var'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = [var_5, var_6, var_7]
    var_9 = 'x'
    var_10 = 'y'
    var_11 = 'z'
    var_12 = [var_9, var_10, var_11]
    var_13 = True
    var_14 = 'key1'
    var_15 = 'key2'
    var_16 = 'value1'
    var_17 = 'value2'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = {var_1: var_8, var_2: var_12, var_3: var_13, var_4: var_18}
    var_20 = {var_0: var_19}
    var_21 = [var_10, var_11]
    var_22 = 'no'
    var_23 = 'key3'
    var_24 = 'new_value2'
    var_25 = 'value3'
    var_26 = {var_15: var_24, var_23: var_25}
    var_27 = {var_1: var_6, var_2: var_21, var_3: var_22, var_4: var_26}
    var_28 = var_20[var_0]
    var_29 = module_0.apply_overwrites_to_context(var_28, var_27)
    var_30 = 'cookiecutter'
    var_31 = var_20[var_30]
    var_32 = 'choice_var'
    var_33 = 'd'
    var_34 = {var_32: var_33}
    var_35 = module_0.apply_overwrites_to_context(var_31, var_34)
    var_36 = 'cookiecutter'
    var_37 = var_20[var_36]
    var_38 = 'multi_choice_var'
    var_39 = 'w'
    var_40 = [var_39]
    var_41 = {var_38: var_40}
    var_42 = module_0.apply_overwrites_to_context(var_37, var_41)
    var_43 = 'cookiecutter'
    var_44 = var_20[var_43]
    var_45 = 'bool_var'
    var_46 = 'maybe'
    var_47 = {var_45: var_46}
    var_48 = module_0.apply_overwrites_to_context(var_44, var_47)



# Parsed testcases at query #2
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp'
    var_7 = module_0.Environment()
    var_8 = '/tmp/test_dir'
    var_9 = ''
    var_10 = {var_2: var_3}
    var_11 = {var_1: var_10}
    var_12 = module_0.Environment()
    var_13 = 'existing_dir'
    var_14 = {var_2: var_3}
    var_15 = {var_1: var_14}
    var_16 = module_0.Environment()
    var_17 = '/tmp/existing_dir'
    var_18 = 'existing_dir'
    var_19 = {var_2: var_3}
    var_20 = {var_1: var_19}
    var_21 = module_0.Environment()
    var_22 = True



# Parsed testcases at query #3
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'Test the render_and_create_dir function.'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp'
    var_7 = module_0.FileSystemLoader(var_6)
    var_8 = module_1.Environment(loader=var_7)
    var_9 = '{{ cookiecutter.project_name }}'
    var_10 = '/tmp/test_project'



# Parsed testcases at query #4
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'Test the generate_file function.'
    var_1 = 'test_project'
    var_2 = 'test_template.txt'
    var_3 = 'cookiecutter'
    var_4 = '_copy_without_render'
    var_5 = '_new_lines'
    var_6 = []
    var_7 = False
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = '.'
    var_11 = module_0.FileSystemLoader(var_10)
    var_12 = module_1.Environment(loader=var_11)
    var_13 = 'Hello {{ name }}!'
    var_14 = module_2.generate_file(var_1, var_2, var_9, var_12)
    var_15 = module_3.rmtree(var_1)



# Parsed testcases at query #5
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'images/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'images/photo.jpg'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True
    var_11 = 'scripts/script.py'
    var_12 = module_0.is_copy_only_path(var_11, var_6)
    assert var_12 is False
    var_13 = 'README.md'
    var_14 = module_0.is_copy_only_path(var_13, var_6)
    assert var_14 is False



# Parsed testcases at query #7
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'Test the generate_file function.'
    var_1 = 'test_project'
    var_2 = 'test_file.txt'
    var_3 = 'cookiecutter'
    var_4 = '_new_lines'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = '.'
    var_9 = module_0.FileSystemLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'Hello, {{ name }}!'
    var_12 = module_2.generate_file(var_1, var_2, var_7, var_10)
    var_13 = module_3.rmtree(var_1)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'Test the generate_files function.'
    var_1 = 'template'
    var_2 = 'output'
    var_3 = '{{cookiecutter.project_name}}.txt'
    var_4 = 'Hello, {{cookiecutter.project_name}}!'
    assert var_4 == 'Hello, TestProject!'
    var_5 = 'cookiecutter.json'
    var_6 = '{"project_name": "TestProject"}'
    var_7 = True
    var_8 = 'TestProject.txt'
    var_9 = var_2 / var_8



# Parsed testcases at query #9
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'test_project_dir'
    var_1 = 'test_infile.txt'
    var_2 = 'cookiecutter'
    var_3 = '_new_lines'
    var_4 = '\n'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'Hello, {{ cookiecutter._new_lines }}'
    assert var_10 == 'Hello, \n'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'Test the generate_file function.'
    var_1 = 'test_template.txt'
    var_2 = 'Hello {{ name }}!'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = 'output'
    var_7 = 'test_binary.bin'
    var_8 = b'\x00\x01\x02\x03'
    var_9 = True



# Parsed testcases at query #11
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'test_project'
    var_1 = 'test_template.txt'
    var_2 = 'name'
    var_3 = 'Test Project'
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = 'Project: {{ name }}'
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7)
    var_10 = module_3.rmtree(var_0)



# Parsed testcases at query #12
#--------------------------


import cookiecutter.utils as module_0

def test_case_0():
    var_0 = 'Test the generate_files function.'
    var_1 = 'tests/fake-repo'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'Test Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'tests/output'
    var_8 = True
    var_9 = False
    var_10 = True
    var_11 = False
    var_12 = 'undefined_var'
    var_13 = '{{ undefined_var }}'
    var_14 = {var_12: var_13}
    var_15 = {var_2: var_14}
    var_16 = False
    var_17 = 'fake-repo'
    var_18 = True
    var_19 = 'fake-repo'
    var_20 = module_0.rmtree(var_1)
    var_21 = 'All tests passed!'
    var_22 = print(var_21)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test the generate_file function.'
    var_1 = 'project'
    var_2 = var_0 / var_1
    var_3 = 'template'
    var_4 = 'test.txt'
    var_5 = 'Hello {{ name }}!'
    var_6 = 'name'
    var_7 = 'World'
    var_8 = {var_6: var_7}
    var_9 = str(var_2)
    var_10 = var_2 / var_4



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'tests/fake-repo-tmpl'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'author_name'
    var_5 = 'email'
    var_6 = 'version'
    var_7 = 'license'
    var_8 = '_copy_without_render'
    var_9 = 'Test Project'
    var_10 = 'test_project'
    var_11 = 'Test Author'
    var_12 = 'test@example.com'
    var_13 = '0.1.0'
    var_14 = 'MIT'
    var_15 = '*.txt'
    var_16 = [var_15]
    var_17 = {var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_12, var_6: var_13, var_7: var_14, var_8: var_16}
    var_18 = {var_1: var_17}
    var_19 = 'tests/fake-repo'
    var_20 = True
    var_21 = False
    var_22 = True
    var_23 = False
    var_24 = 'README.md'
    var_25 = 'LICENSE'
    var_26 = 'requirements.txt'



# Parsed testcases at query #15
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'name'
    var_2 = 'default_name'
    var_3 = {var_1: var_2}
    var_4 = 'extra_name'
    var_5 = {var_1: var_4}
    var_6 = 'name'
    var_7 = 'test_name'
    var_8 = {var_6: var_7}
    var_9 = module_0.generate_context(var_0, var_3)
    var_10 = module_0.generate_context(var_0, var_3, var_5)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'template.txt'
    var_1 = 'Hello, {{ name }}!'
    var_2 = 'project'
    var_3 = 'name'
    var_4 = 'Cookiecutter'
    var_5 = {var_3: var_4}



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test the generate_file function.'
    var_1 = 'test_dir'
    var_2 = True
    var_3 = 'test_file.txt'
    var_4 = 'Hello {{ name }}!'
    var_5 = 'utf-8'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = '_new_lines'
    var_9 = 'World'
    var_10 = False
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = {var_6: var_11}
    var_13 = 'output_dir'



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'Test the generate_files function.'
    var_1 = 'tests/fake-repo'
    var_2 = 'cookiecutter.json'
    var_3 = 'tests/output'
    var_4 = True
    var_5 = 'README.rst'



# Parsed testcases at query #19
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test the render_and_create_dir function.'
    var_1 = 'test_dir'
    var_2 = {}
    var_3 = module_0.Environment()
    var_4 = ''
    var_5 = 'existing_dir'
    var_6 = True
    var_7 = 'another_existing_dir'
    var_8 = False



# Parsed testcases at query #20
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'test.json'
    var_4 = module_0.generate_context(var_3)
    var_5 = 'invalid json'
    var_6 = 'test.json'
    var_7 = module_0.generate_context(var_6)
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'key'
    var_12 = 'default'
    var_13 = {var_11: var_12}
    var_14 = module_0.generate_context(var_8, var_13)
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = 'extra'
    var_19 = {var_11: var_18}
    var_20 = module_0.generate_context(var_15, extra_context=var_19)



# Parsed testcases at query #21
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/project_dir'
    var_1 = 'template.txt'
    var_2 = 'variable'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = '{{ variable }}'
    var_10 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)



# Parsed testcases at query #22
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-context/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'non-existent.json'
    var_3 = module_0.generate_context(var_2)
    var_4 = 'tests/test-context/invalid.json'
    var_5 = module_0.generate_context(var_4)
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'Default Project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = module_0.generate_context(var_0, var_10)
    var_12 = 'Extra Project'
    var_13 = {var_7: var_12}
    var_14 = {var_6: var_13}
    var_15 = module_0.generate_context(var_0, extra_context=var_14)



# Parsed testcases at query #23
#--------------------------


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
    var_8 = False
    var_9 = True
    var_10 = False
    var_11 = module_0.generate_files(var_0, var_5, var_6, var_7, var_8, var_9, var_10)
    assert var_11 == 'test_output/test_project'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'Test the generate_files function.'
    var_1 = 'tests/fake-repo-pre'
    var_2 = 'cookiecutter.json'
    var_3 = 'tests/output'
    var_4 = True
    var_5 = False
    var_6 = True
    var_7 = False
    var_8 = 'README.rst'
    var_9 = 'hooks'
    var_10 = 'post_gen_project.py'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test the generate_file function.'
    var_1 = 'test_template.txt'
    var_2 = 'Hello {{ name }}!'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = 'project'



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'description'
    var_3 = '_copy_without_render'
    var_4 = 'Test Project'
    var_5 = '1.0'
    var_6 = 'A test project'
    var_7 = '*.txt'
    var_8 = [var_7]
    var_9 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_8}
    var_10 = '2.0'
    var_11 = {var_1: var_10}
    var_12 = 'Overridden description'
    var_13 = {var_2: var_12}
    var_14 = 'invalid json'
    var_15 = 'All tests passed!'
    var_16 = print(var_15)



# Parsed testcases at query #27
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'test_project'
    var_1 = 'template.txt'
    var_2 = 'variable'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = False
    var_9 = '{{ variable }}'
    var_10 = module_2.generate_file(var_0, var_1, var_4, var_7, var_8)
    var_11 = module_3.rmtree(var_0)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'project_slug'
    var_3 = 'Test Project'
    var_4 = 'test_project'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'Project Name: {{ cookiecutter.project_name }}'
    var_7 = 'output'
    var_8 = 'README.md'



# Parsed testcases at query #29
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'non-existent.json'
    var_3 = module_0.generate_context(var_2)
    var_4 = 'cookiecutter'
    var_5 = 'project_name'
    var_6 = 'Default Project'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'Extra Project'
    var_10 = {var_5: var_9}
    var_11 = {var_4: var_10}
    var_12 = module_0.generate_context(var_0, var_8, var_11)
    var_13 = 'tests/invalid-context.json'
    var_14 = module_0.generate_context(var_13)
    var_15 = 'All tests passed!'
    var_16 = print(var_15)



# Parsed testcases at query #30
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test the generate_context function.'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'Test Project'
    var_4 = '0.1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.generate_context()
    var_7 = 'project_name'
    var_8 = 'Overridden Project'
    var_9 = {var_7: var_8}
    var_10 = module_0.generate_context(extra_context=var_9)
    var_11 = 'cookiecutter.json'



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'invalid json'
    var_4 = 'default_key'
    var_5 = 'default_value'
    var_6 = {var_4: var_5}
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'extra_key'
    var_11 = 'extra_value'
    var_12 = {var_10: var_11}
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = 'All test cases passed'
    var_17 = print(var_16)



# Parsed testcases at query #32
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = module_0.generate_context()
    var_1 = 'custom_context.json'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_1)
    var_6 = 'key'
    var_7 = 'default_value'
    var_8 = {var_6: var_7}
    var_9 = module_0.generate_context(default_context=var_8)
    var_10 = 'extra_value'
    var_11 = {var_6: var_10}
    var_12 = module_0.generate_context(extra_context=var_11)
    var_13 = 'invalid.json'
    var_14 = 'invalid json'
    var_15 = module_0.generate_context(var_13)



# Parsed testcases at query #33
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'My Project'
    var_2 = {var_0: var_1}
    var_3 = module_0.generate_context()
    var_4 = 'project_name'
    var_5 = 'New Project'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(extra_context=var_6)
    var_8 = 'cookiecutter.json'



# Parsed testcases at query #34
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-context.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'Default Project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'Extra Project'
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = module_0.generate_context(var_0, var_5, var_8)



# Parsed testcases at query #35
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'Test the generate_file function.'
    var_1 = 'test_project'
    var_2 = 'test_template.txt'
    var_3 = 'cookiecutter'
    var_4 = '_copy_without_render'
    var_5 = '_new_lines'
    var_6 = []
    var_7 = False
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = '.'
    var_11 = module_0.FileSystemLoader(var_10)
    var_12 = module_1.Environment(loader=var_11)
    var_13 = 'Hello {{ name }}!'
    var_14 = module_2.generate_file(var_1, var_2, var_9, var_12)
    var_15 = module_3.rmtree(var_1)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'variable'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'output'
    var_4 = module_0.Environment()
    var_5 = 'new_dir'
    var_6 = 'existing_dir'
    var_7 = True
    var_8 = 'existing_dir_no_overwrite'
    var_9 = ''



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'project'
    var_2 = 'test_file.txt'
    var_3 = 'Hello, {{ name }}!'
    var_4 = 'name'
    var_5 = 'World'
    var_6 = {var_4: var_5}



# Parsed testcases at query #3
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'use_docker'
    var_3 = 'features'
    var_4 = 'database'
    var_5 = 'My Project'
    var_6 = True
    var_7 = 'feature1'
    var_8 = 'feature2'
    var_9 = [var_7, var_8]
    var_10 = 'engine'
    var_11 = 'port'
    var_12 = 'postgres'
    var_13 = 5432
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_14}
    var_16 = {var_0: var_15}
    var_17 = 'New Project'
    var_18 = 'no'
    var_19 = 'feature3'
    var_20 = [var_8, var_19]
    var_21 = 3306
    var_22 = {var_11: var_21}
    var_23 = {var_1: var_17, var_2: var_18, var_3: var_20, var_4: var_22}
    var_24 = var_16[var_0]
    var_25 = module_0.apply_overwrites_to_context(var_24, var_23)



# Parsed testcases at query #4
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/fake-repo-pre'
    var_1 = 'tests/output'
    var_2 = 'tests/fake-repo-pre/cookiecutter.json'
    var_3 = module_0.generate_context(var_2)
    var_4 = True



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'tests/test-repo-pre/'
    var_1 = 'tests/test-output/'
    var_2 = True
    var_3 = True
    var_4 = True
    var_5 = 'tests/test-repo-undef-var/'
    var_6 = 'tests/test-repo-empty-dir/'
    var_7 = 'tests/test-repo-invalid-json/'



# Parsed testcases at query #6
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test the generate_context function.'
    var_1 = 'tests/test-context.json'
    var_2 = module_0.generate_context(var_1)
    var_3 = 'non-existent.json'
    var_4 = module_0.generate_context(var_3)
    var_5 = 'tests/invalid-context.json'
    var_6 = module_0.generate_context(var_5)
    var_7 = 'project_name'
    var_8 = 'Default Project'
    var_9 = {var_7: var_8}
    var_10 = module_0.generate_context(var_1, var_9)
    var_11 = 'Extra Project'
    var_12 = {var_7: var_11}
    var_13 = module_0.generate_context(var_1, extra_context=var_12)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'Test the generate_files function.'
    var_1 = 'cookiecutter.json'
    var_2 = var_0 / var_1
    var_3 = '{"project_name": "Test Project", "repo_name": "{{ cookiecutter.project_name.lower().replace(\' \', \'-\') }}"}'
    var_4 = 'README.md'
    var_5 = '# {{ cookiecutter.project_name }}'
    var_6 = 'static'
    var_7 = 'logo.png'
    var_8 = 'PNG image data'
    var_9 = 'cookiecutter'
    var_10 = 'project_name'
    var_11 = 'Test Project'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = True
    var_15 = 'README.md'
    var_16 = 'static'
    var_17 = var_6 / var_16
    var_18 = 'logo.png'
    var_19 = var_17 / var_18
    var_20 = 'existing_file'
    var_21 = 'original'
    var_22 = False
    var_23 = 'hooks'
    var_24 = 'pre_gen_project.py'
    var_25 = 'print("Pre-gen hook")'
    var_26 = 'post_gen_project.py'
    var_27 = 'print("Post-gen hook")'



# Parsed testcases at query #8
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, var_3, var_6)
    var_8 = 'invalid_cookiecutter.json'
    var_9 = module_0.generate_context(var_8)
    var_10 = 'test_cookiecutter.json'
    var_11 = 'invalid_value'
    var_12 = {var_9: var_11}
    var_13 = module_0.generate_context(var_10, var_12)
    var_14 = {var_4: var_11}
    var_15 = None
    var_16 = module_0.generate_context(var_10, var_15, var_14)



# Parsed testcases at query #9
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'Test the generate_file function.'
    var_1 = 'test_project'
    var_2 = 'test_template.txt'
    var_3 = 'variable'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = '.'
    var_7 = module_0.FileSystemLoader(var_6)
    var_8 = module_1.Environment(loader=var_7)
    var_9 = 'This is a test template with {{ variable }}'
    var_10 = module_2.generate_file(var_1, var_2, var_5, var_8)
    var_11 = module_3.rmtree(var_1)



# Parsed testcases at query #10
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'Test the generate_file function.'
    var_1 = 'test_project'
    var_2 = 'test_template.txt'
    var_3 = 'cookiecutter'
    var_4 = '_new_lines'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = '.'
    var_9 = module_0.FileSystemLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'Hello {{ name }}!'
    var_12 = module_2.generate_file(var_1, var_2, var_7, var_10)
    var_13 = module_3.rmtree(var_1)



# Parsed testcases at query #11
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'images/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'example.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'images/photo.jpg'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True
    var_11 = 'scripts/script.py'
    var_12 = module_0.is_copy_only_path(var_11, var_6)
    assert var_12 is False
    var_13 = 'README.md'
    var_14 = module_0.is_copy_only_path(var_13, var_6)
    assert var_14 is False



# Parsed testcases at query #12
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'Test the is_copy_only_path function.'
    var_1 = 'cookiecutter'
    var_2 = '_copy_without_render'
    var_3 = '*.txt'
    var_4 = 'docs/*'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'file.txt'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True
    var_10 = 'docs/file.txt'
    var_11 = module_0.is_copy_only_path(var_10, var_7)
    assert var_11 is True
    var_12 = 'docs/subdir/file.txt'
    var_13 = module_0.is_copy_only_path(var_12, var_7)
    assert var_13 is True
    var_14 = 'file.py'
    var_15 = module_0.is_copy_only_path(var_14, var_7)
    assert var_15 is False
    var_16 = 'templates/file.txt'
    var_17 = module_0.is_copy_only_path(var_16, var_7)
    assert var_17 is False
    var_18 = {}
    var_19 = module_0.is_copy_only_path(var_8, var_18)
    assert var_19 is False



# Parsed testcases at query #13
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'Test generate_file function.'
    var_1 = 'test_project'
    var_2 = 'test_template.txt'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'Test Project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = '.'
    var_9 = module_0.FileSystemLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'Project Name: {{ cookiecutter.project_name }}'
    var_12 = 'test_template.txt'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'TestProject'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '{{ cookiecutter.project_name }}'
    var_6 = True



# Parsed testcases at query #15
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test the render_and_create_dir function.'
    var_1 = 'test_dir'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_output'
    var_8 = module_0.Environment()
    var_9 = False
    var_10 = True
    var_11 = False
    var_12 = ''



# Parsed testcases at query #16
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/unit/test_context.json'
    var_1 = 'project_name'
    var_2 = 'Default Project'
    var_3 = {var_1: var_2}
    var_4 = 'Extra Project'
    var_5 = {var_1: var_4}
    var_6 = module_0.generate_context(var_0, var_3, var_5)
    var_7 = 'nonexistent.json'
    var_8 = module_0.generate_context(var_7)
    var_9 = 'tests/unit/invalid.json'
    var_10 = module_0.generate_context(var_9)
    var_11 = module_0.generate_context(var_0)
    var_12 = 'invalid_key'
    var_13 = 'Invalid Value'
    var_14 = {var_12: var_13}
    var_15 = module_0.generate_context(var_0, var_14)
    var_16 = None
    var_17 = 'invalid_key'
    var_18 = 'Invalid Value'
    var_19 = {var_17: var_18}
    var_20 = module_0.generate_context(var_0, var_16, var_19)
    var_21 = {var_16: var_17}
    var_22 = module_0.generate_context(var_0, var_21)
    var_23 = None
    var_24 = {var_16: var_18}
    var_25 = module_0.generate_context(var_0, var_23, var_24)
    var_26 = {var_16: var_17}
    var_27 = {var_16: var_18}
    var_28 = module_0.generate_context(var_0, var_26, var_27)
    var_29 = 'tests/unit/empty.json'
    var_30 = module_0.generate_context(var_29)



# Parsed testcases at query #17
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test the render_and_create_dir function.'
    var_1 = 'test_dir'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_output'
    var_8 = module_0.Environment()
    var_9 = False
    var_10 = True
    var_11 = ''



# Parsed testcases at query #18
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(default_context=var_2, extra_context=var_5)
    var_7 = 'invalid.json'
    var_8 = module_0.generate_context(var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 'valid.json'
    var_13 = module_0.generate_context(var_12)



# Parsed testcases at query #19
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'features'
    var_4 = 'settings'
    var_5 = 'My Project'
    var_6 = '1.0'
    var_7 = 'feature1'
    var_8 = 'feature2'
    var_9 = [var_7, var_8]
    var_10 = 'debug'
    var_11 = 'log_level'
    var_12 = True
    var_13 = 'info'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = {var_1: var_5, var_2: var_6, var_3: var_9, var_4: var_14}
    var_16 = {var_0: var_15}
    var_17 = 'Overwritten Project'
    var_18 = 'feature3'
    var_19 = [var_18]
    var_20 = {var_11: var_10}
    var_21 = {var_1: var_17, var_3: var_19, var_4: var_20}
    var_22 = module_0.apply_overwrites_to_context(var_16, var_21)



# Parsed testcases at query #20
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/fixtures/fake-repo/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'cookiecutter'
    var_3 = 'full_name'
    var_4 = 'Default Name'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.generate_context(var_0, var_6)
    var_8 = 'Extra Name'
    var_9 = {var_3: var_8}
    var_10 = {var_2: var_9}
    var_11 = module_0.generate_context(var_0, extra_context=var_10)
    var_12 = 'nonexistent.json'
    var_13 = module_0.generate_context(var_12)
    var_14 = 'tests/fixtures/invalid-repo/cookiecutter.json'
    var_15 = module_0.generate_context(var_14)



# Parsed testcases at query #21
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/fake-repo/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'Test Project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'Extra Project'
    var_7 = {var_2: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = 'tests/fake-repo/invalid.json'
    var_10 = module_0.generate_context(var_9)
    var_11 = 'tests/fake-repo/invalid-json.json'
    var_12 = module_0.generate_context(var_11)



# Parsed testcases at query #22
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = '/tmp'
    var_6 = '.'
    var_7 = [var_6]
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = '{{ cookiecutter.name }}'
    var_11 = '/tmp/test_project'
    var_12 = ''
    var_13 = '{{ cookiecutter.name }}'
    var_14 = '{{ cookiecutter.name }}'
    var_15 = True



# Parsed testcases at query #23
#--------------------------


import cookiecutter.generate as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'TestProject'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)
    var_8 = module_1.rmtree(var_6)
    var_9 = True
    var_10 = module_0.generate_files(var_0, var_5, var_6, var_9)
    var_11 = module_1.rmtree(var_6)
    var_12 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_9)
    var_13 = module_1.rmtree(var_6)
    var_14 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_9, keep_project_on_failure=var_9)
    var_15 = module_1.rmtree(var_6)
    var_16 = '{{ undefined_variable }}'
    var_17 = {var_2: var_16}
    var_18 = {var_1: var_17}
    var_19 = module_0.generate_files(var_0, var_18, var_6)
    var_20 = module_1.rmtree(var_6)



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'template.txt'
    var_1 = 'Hello, {{ name }}!'
    var_2 = 'name'
    var_3 = 'World'
    var_4 = {var_2: var_3}



# Parsed testcases at query #25
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'test_project'
    var_1 = 'template.txt'
    var_2 = 'variable'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = '.'
    var_6 = module_0.FileSystemLoader(var_5)
    var_7 = module_1.Environment(loader=var_6)
    var_8 = 'This is a template with {{ variable }}.'
    var_9 = module_2.generate_file(var_0, var_1, var_4, var_7)
    var_10 = module_3.rmtree(var_0)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'Hello {{ name }}'
    var_1 = 'name'
    assert var_1 == 'Hello World'
    var_2 = 'World'
    var_3 = {var_1: var_2}
    var_4 = 'test.txt'



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 'Test the generate_files function.'
    var_1 = 'tests'
    var_2 = 'fake-repo-tmpl'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'fake-project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 'fake-output-dir'
    var_9 = True
    var_10 = False
    var_11 = True
    var_12 = False



# Parsed testcases at query #28
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'invalid json'
    var_6 = module_0.generate_context(var_0)
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'key'
    var_11 = 'new_value'
    var_12 = {var_10: var_11}
    var_13 = module_0.generate_context(var_0, var_12)
    var_14 = 'key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = 'extra_value'
    var_18 = {var_15: var_17}
    var_19 = module_0.generate_context(var_0, extra_context=var_18)



# Parsed testcases at query #29
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'Test Project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'This is a test template for {{ cookiecutter.name }}'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_12 = module_3.rmtree(var_0)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'tests/fake-repo-pre'
    var_1 = 'tests/output'
    var_2 = 'cookiecutter'
    var_3 = 'full_name'
    var_4 = 'email'
    var_5 = 'project_name'
    var_6 = 'repo_name'
    var_7 = '_copy_without_render'
    var_8 = 'Test User'
    var_9 = 'test@example.com'
    var_10 = 'Test Project'
    var_11 = 'test_project'
    var_12 = []
    var_13 = {var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11, var_7: var_12}
    var_14 = {var_2: var_13}
    var_15 = True
    var_16 = False
    var_17 = True
    var_18 = False
    var_19 = []
    var_20 = {var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11, var_7: var_19}
    var_21 = {var_2: var_20}
    var_22 = False
    var_23 = False
    var_24 = True
    var_25 = False
    var_26 = []
    var_27 = {var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11, var_7: var_26}
    var_28 = {var_2: var_27}
    var_29 = True
    var_30 = True
    var_31 = True
    var_32 = False
    var_33 = []
    var_34 = {var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11, var_7: var_33}
    var_35 = {var_2: var_34}
    var_36 = True
    var_37 = False
    var_38 = False
    var_39 = False
    var_40 = []
    var_41 = {var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11, var_7: var_40}
    var_42 = {var_2: var_41}
    var_43 = True
    var_44 = False
    var_45 = True
    var_46 = True
    var_47 = 'undefined_variable'
    var_48 = []
    var_49 = '{{ undefined_variable }}'
    var_50 = {var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11, var_7: var_48, var_47: var_49}
    var_51 = {var_2: var_50}



# Parsed testcases at query #31
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'non-existent.json'
    var_3 = module_0.generate_context(var_2)
    var_4 = 'tests/invalid-context.json'
    var_5 = module_0.generate_context(var_4)
    var_6 = 'project_name'
    var_7 = 'Default Project'
    var_8 = {var_6: var_7}
    var_9 = module_0.generate_context(var_0, var_8)
    var_10 = 'Extra Project'
    var_11 = {var_6: var_10}
    var_12 = None
    var_13 = module_0.generate_context(var_0, var_12, var_11)
    var_14 = 'All tests passed!'
    var_15 = print(var_14)



# Parsed testcases at query #32
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = 'Test the generate_file function.'
    var_1 = 'test_project'
    var_2 = 'test_template.txt'
    var_3 = 'cookiecutter'
    var_4 = '_copy_without_render'
    var_5 = '_new_lines'
    var_6 = []
    var_7 = False
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}
    var_10 = '.'
    var_11 = module_0.FileSystemLoader(var_10)
    var_12 = module_1.Environment(loader=var_11)
    var_13 = 'Hello {{ name }}!'
    var_14 = module_2.generate_file(var_1, var_2, var_9, var_12)
    var_15 = module_3.rmtree(var_1)



