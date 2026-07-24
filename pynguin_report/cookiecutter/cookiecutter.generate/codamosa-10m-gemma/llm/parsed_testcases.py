####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'old_name'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_name'
    var_6 = 2
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = 'existing'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 'new_var'
    var_13 = 'should_not_appear'
    var_14 = {var_12: var_13}
    var_15 = False
    var_16 = module_0.apply_overwrites_to_context(var_11, var_14, in_dictionary_variable=var_15)
    var_17 = 'nested'
    var_18 = 'a'
    var_19 = {var_18: var_3}
    var_20 = {var_17: var_19}
    var_21 = 'b'
    var_22 = {var_21: var_6}
    var_23 = {var_17: var_22}
    var_24 = True
    var_25 = module_0.apply_overwrites_to_context(var_20, var_23, in_dictionary_variable=var_24)
    var_26 = {var_18: var_24, var_21: var_6}
    var_27 = {var_17: var_26}
    var_28 = 3
    var_29 = {var_21: var_28}
    var_30 = {var_17: var_29}
    var_31 = True
    var_32 = module_0.apply_overwrites_to_context(var_27, var_30, in_dictionary_variable=var_31)
    var_33 = 'choices'
    var_34 = 'c'
    var_35 = [var_18, var_21, var_34]
    var_36 = {var_33: var_35}
    var_37 = [var_18, var_34]
    var_38 = {var_33: var_37}
    var_39 = module_0.apply_overwrites_to_context(var_36, var_38)
    var_40 = [var_18, var_21, var_34]
    var_41 = {var_33: var_40}
    var_42 = 'z'
    var_43 = [var_18, var_42]
    var_44 = {var_33: var_43}
    var_45 = module_0.apply_overwrites_to_context(var_41, var_44)
    var_46 = 'choice'
    var_47 = [var_18, var_21, var_34]
    var_48 = {var_46: var_47}
    var_49 = {var_46: var_21}
    var_50 = module_0.apply_overwrites_to_context(var_48, var_49)
    var_51 = [var_18, var_21, var_34]
    var_52 = {var_46: var_51}
    var_53 = {var_46: var_42}
    var_54 = module_0.apply_overwrites_to_context(var_52, var_53)
    var_55 = 'is_enabled'
    var_56 = {var_55: var_15}
    var_57 = 'yes'
    var_58 = {var_55: var_57}
    var_59 = module_0.apply_overwrites_to_context(var_56, var_58)
    var_60 = True
    var_61 = {var_55: var_60}
    var_62 = 'no'
    var_63 = {var_55: var_62}
    var_64 = module_0.apply_overwrites_to_context(var_61, var_63)
    var_65 = True
    var_66 = {var_55: var_65}
    var_67 = 'not_a_boolean'
    var_68 = {var_55: var_67}
    var_69 = module_0.apply_overwrites_to_context(var_66, var_68)



# Parsed testcases at query #2
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test the render_and_create_dir function with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = '{{ project_name }}_dir'
    var_7 = False
    var_8 = ''
    var_9 = False
    var_10 = str(var_9)
    var_11 = 'old_file.txt'
    var_12 = 'old content'
    var_13 = '{{ project_name }}_dir'
    var_14 = True
    var_15 = 'project_{{ project_name }}_{{ project_name }}'



# Parsed testcases at query #3
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Tests the render_and_create_dir function with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'output'
    var_3 = 'project_name'
    var_4 = 'my_project'
    var_5 = {var_3: var_4}
    var_6 = '{{ project_name }}_dir'
    var_7 = False
    var_8 = ''
    var_9 = 'already_exists'
    var_10 = 'already_exists'
    var_11 = False
    var_12 = 'overwrite_me'
    var_13 = True
    var_14 = 'version'
    var_15 = 'complex'
    var_16 = '1.0'
    var_17 = {var_3: var_15, var_14: var_16}
    var_18 = '{{ project_name }}_v{{ version }}'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test the generate_file function with various scenarios.'
    var_1 = 'template_dir'
    var_2 = 'output_dir'
    var_3 = 'context'
    var_4 = 'env'
    var_5 = 'hello_{{ name }}.txt'
    var_6 = 'hello_World.txt'
    var_7 = 'utf-8'
    var_8 = 'hello_{{ name }}.txt'
    var_9 = 'hello_Tester.txt'
    var_10 = 'data.bin'
    var_11 = 'existing.txt'
    var_12 = 'Original Content'
    var_13 = 'collision.txt'
    var_14 = 'New Content'
    var_15 = True
    var_16 = False

def test_case_0():
    var_0 = 'Test that generate_file raises TemplateSyntaxError for invalid templates.'
    var_1 = 'template_dir'
    var_2 = 'output_dir'
    var_3 = 'env'
    var_4 = 'context'
    var_5 = 'broken.txt'
    var_6 = 'Hello {{ name'
    var_7 = 'utf-8'
    var_8 = 'broken.txt'



# Parsed testcases at query #5
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old_name'
    var_3 = 1.0
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'author'
    var_6 = 'new_name'
    var_7 = 'tester'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = 'choices'
    var_11 = 'a'
    var_12 = 'b'
    var_13 = 'c'
    var_14 = [var_11, var_12, var_13]
    var_15 = {var_10: var_14}
    var_16 = [var_11, var_13]
    var_17 = {var_10: var_16}
    var_18 = module_0.apply_overwrites_to_context(var_15, var_17)
    var_19 = [var_11, var_12, var_13]
    var_20 = {var_10: var_19}
    var_21 = 'z'
    var_22 = [var_11, var_21]
    var_23 = {var_10: var_22}
    var_24 = module_0.apply_overwrites_to_context(var_20, var_23)
    var_25 = 'choice'
    var_26 = [var_11, var_12, var_13]
    var_27 = {var_25: var_26}
    var_28 = {var_25: var_13}
    var_29 = module_0.apply_overwrites_to_context(var_27, var_28)
    var_30 = 'nested'
    var_31 = 'key1'
    var_32 = 'key2'
    var_33 = 'val1'
    var_34 = 'val2'
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = {var_30: var_35}
    var_37 = 'key3'
    var_38 = 'new_val2'
    var_39 = 'val3'
    var_40 = {var_32: var_38, var_37: var_39}
    var_41 = {var_30: var_40}
    var_42 = module_0.apply_overwrites_to_context(var_36, var_41)
    var_43 = 'enabled'
    var_44 = False
    var_45 = {var_43: var_44}
    var_46 = 'yes'
    var_47 = {var_43: var_46}
    var_48 = module_0.apply_overwrites_to_context(var_45, var_47)
    var_49 = {var_43: var_44}
    var_50 = 'not-a-boolean'
    var_51 = {var_43: var_50}
    var_52 = module_0.apply_overwrites_to_context(var_49, var_51)
    var_53 = 'outer'
    var_54 = 'inner_list'
    var_55 = [var_11, var_12]
    var_56 = {var_54: var_55}
    var_57 = {var_53: var_56}
    var_58 = [var_13]
    var_59 = {var_54: var_58}
    var_60 = {var_53: var_59}
    var_61 = True
    var_62 = module_0.apply_overwrites_to_context(var_57, var_60, in_dictionary_variable=var_61)
    var_63 = 'existing'
    var_64 = 'val'
    var_65 = {var_63: var_64}
    var_66 = 'new_key'
    var_67 = 'ignore_me'
    var_68 = {var_66: var_67}
    var_69 = module_0.apply_overwrites_to_context(var_65, var_68, in_dictionary_variable=var_44)
    var_70 = 'options'
    var_71 = 'apple'
    var_72 = 'banana'
    var_73 = [var_71, var_72]
    var_74 = {var_70: var_73}
    var_75 = {var_70: var_72}
    var_76 = module_0.apply_overwrites_to_context(var_74, var_75)



# Parsed testcases at query #6
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test generate_context with various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'version'
    var_5 = 'my_project'
    var_6 = 'test_user'
    var_7 = '0.1.0'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'new_var'
    var_11 = 'new_author'
    var_12 = 'new_value'
    var_13 = {var_3: var_11, var_10: var_12}
    var_14 = 'default_project'
    var_15 = {var_2: var_14}
    var_16 = 'invalid.json'
    var_17 = "{ 'broken': json }"
    var_18 = module_1.generate_context(var_0)
    var_19 = 'does_not_exist.json'
    var_20 = module_1.generate_context(var_0)
    var_21 = 'my_template.json'
    var_22 = 'key'
    var_23 = 'val'
    var_24 = {var_22: var_23}
    var_25 = module_0.dumps(var_24)

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Test that warnings are issued when default_context contains invalid overrides.'
    var_1 = 'cookiecutter.json'
    var_2 = 'choice_var'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = 'c'
    var_9 = [var_8]
    var_10 = {var_2: var_9}
    var_11 = module_1.generate_context(var_0, var_10)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'template.txt'
    var_1 = 'Hello {{ project_name }}'
    var_2 = 'output'

def test_case_0():
    var_0 = 'binary.dat'
    var_1 = b'\x00\x01\x02\x03'
    assert var_1 == b'\x00\x01\x02\x03'
    var_2 = 'output'

def test_case_0():
    var_0 = 'exists.txt'
    var_1 = 'original'
    var_2 = 'output'
    var_3 = "don't overwrite me"
    assert var_3 == "don't overwrite me"
    var_4 = True

def test_case_0():
    var_0 = 'template.txt'
    var_1 = 'content'
    var_2 = 'output'
    var_3 = 'my_project_file.txt'
    var_4 = 'template_{{ project_name }}.txt'



# Parsed testcases at query #8
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = '\n    Tests that generate_files correctly renders templates, \n    copies files without rendering, and creates the expected directory structure.\n    '
    var_1 = 'repo_dir'
    var_2 = 'output_dir'
    var_3 = 'context'
    var_4 = None
    var_5 = module_0.Environment(loader=var_4)
    var_6 = True
    var_7 = False
    var_8 = 'README.md'
    var_9 = 'utf-8'
    var_10 = 'static_assets'
    var_11 = 'data.txt'

def test_case_0():
    var_0 = 'Tests that UndefinedVariableInTemplate is raised when a variable is missing.'
    var_1 = 'repo_dir'
    var_2 = 'output_dir'
    var_3 = 'cookiecutter'
    var_4 = 'project_name'
    var_5 = 'fail_project'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = False



# Parsed testcases at query #9
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new'
    var_6 = 2
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = {var_0: var_2}
    var_10 = 'new_var'
    var_11 = 'should_not_be_here'
    var_12 = {var_10: var_11}
    var_13 = False
    var_14 = module_0.apply_overwrites_to_context(var_9, var_12, in_dictionary_variable=var_13)
    var_15 = 'nested'
    var_16 = 'a'
    var_17 = {var_16: var_3}
    var_18 = {var_15: var_17}
    var_19 = 'b'
    var_20 = {var_19: var_6}
    var_21 = {var_15: var_20}
    var_22 = True
    var_23 = 'settings'
    var_24 = 'theme'
    var_25 = 'font'
    var_26 = 'dark'
    var_27 = 'serif'
    var_28 = {var_24: var_26, var_25: var_27}
    var_29 = {var_23: var_28}
    var_30 = 'light'
    var_31 = {var_24: var_30}
    var_32 = {var_23: var_31}
    var_33 = True
    var_34 = module_0.apply_overwrites_to_context(var_29, var_32, in_dictionary_variable=var_33)
    var_35 = 'choices'
    var_36 = 'c'
    var_37 = [var_16, var_19, var_36]
    var_38 = {var_35: var_37}
    var_39 = [var_16, var_36]
    var_40 = {var_35: var_39}
    var_41 = module_0.apply_overwrites_to_context(var_38, var_40)
    var_42 = [var_16, var_19, var_0]
    var_43 = {var_35: var_42}
    var_44 = 'z'
    var_45 = [var_44]
    var_46 = {var_35: var_45}
    var_47 = module_0.apply_overwrites_to_context(var_43, var_46)
    var_48 = 'choice'
    var_49 = 'option1'
    var_50 = 'option2'
    var_51 = [var_49, var_50]
    var_52 = {var_48: var_51}
    var_53 = {var_48: var_50}
    var_54 = module_0.apply_overwrites_to_context(var_52, var_53)
    var_55 = [var_49, var_50]
    var_56 = {var_48: var_55}
    var_57 = 'option3'
    var_58 = {var_48: var_57}
    var_59 = module_0.apply_overwrites_to_context(var_56, var_58)
    var_60 = 'enabled'
    var_61 = {var_60: var_13}
    var_62 = 'yes'
    var_63 = {var_60: var_62}
    var_64 = module_0.apply_overwrites_to_context(var_61, var_63)
    var_65 = True
    var_66 = {var_60: var_65}
    var_67 = 'no'
    var_68 = {var_60: var_67}
    var_69 = module_0.apply_overwrites_to_context(var_66, var_68)
    var_70 = {var_60: var_13}
    var_71 = 'not_a_boolean'
    var_72 = {var_60: var_71}
    var_73 = module_0.apply_overwrites_to_context(var_70, var_72)
    var_74 = 'list_var'
    var_75 = [var_16, var_19]
    var_76 = {var_74: var_75}
    var_77 = {var_74: var_16}
    var_78 = module_0.apply_overwrites_to_context(var_76, var_77)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'hello_{{ name }}.txt'
    var_1 = 'hello_{{ name }}.txt'
    var_2 = 'Hello, {{ name }}!'
    var_3 = 'utf-8'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = '_new_lines'
    var_7 = '\n'
    var_8 = {var_6: var_7}
    var_9 = 'World'
    var_10 = {var_4: var_8, var_5: var_9}
    var_11 = False
    var_12 = 'hello_World.txt'

def test_case_0():
    var_0 = 'Test that binary files are copied without rendering.'
    var_1 = 'data.bin'
    var_2 = 'data.bin'
    var_3 = b'\x00\x01\x02\x03'
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'Test that file is not overwritten if skip_if_file_exists is True.'
    var_1 = 'existing.txt'
    var_2 = 'existing.txt'
    var_3 = 'New Content'
    var_4 = 'utf-8'
    var_5 = 'Old Content'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = {}
    var_9 = 'Test'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = True

def test_case_0():
    var_0 = 'Test that TemplateSyntaxError is raised correctly.'
    var_1 = 'error.txt'
    var_2 = 'error.txt'
    var_3 = 'Hello {{ name '
    var_4 = 'utf-8'
    var_5 = 'cookiecutter'
    var_6 = {}
    var_7 = {var_5: var_6}



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'old_name'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_name'
    var_6 = 2
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0.apply_overwrites_to_context(var_4, var_7)
    var_9 = 'existing'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 'new_var'
    var_13 = 'ignored'
    var_14 = {var_12: var_13}
    var_15 = module_0.apply_overwrites_to_context(var_11, var_14)
    var_16 = 'settings'
    var_17 = 'theme'
    var_18 = 'light'
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = 'font'
    var_22 = 'serif'
    var_23 = {var_21: var_22}
    var_24 = {var_16: var_23}
    var_25 = True
    var_26 = module_0.apply_overwrites_to_context(var_20, var_24, in_dictionary_variable=var_25)
    var_27 = 'choices'
    var_28 = 'a'
    var_29 = 'b'
    var_30 = 'c'
    var_31 = [var_28, var_29, var_30]
    var_32 = {var_27: var_31}
    var_33 = [var_28, var_30]
    var_34 = {var_27: var_33}
    var_35 = module_0.apply_overwrites_to_context(var_32, var_34)
    var_36 = [var_28, var_29]
    var_37 = {var_27: var_36}
    var_38 = 'z'
    var_39 = [var_28, var_38]
    var_40 = {var_27: var_39}
    var_41 = module_0.apply_overwrites_to_context(var_37, var_40)
    var_42 = 'choice'
    var_43 = [var_28, var_29, var_30]
    var_44 = {var_42: var_43}
    var_45 = {var_42: var_29}
    var_46 = module_0.apply_overwrites_to_context(var_44, var_45)
    var_47 = [var_28, var_29]
    var_48 = {var_42: var_47}
    var_49 = {var_42: var_38}
    var_50 = module_0.apply_overwrites_to_context(var_48, var_49)
    var_51 = 'debug'
    var_52 = False
    var_53 = {var_51: var_52}
    var_54 = 'yes'
    var_55 = {var_51: var_54}
    var_56 = module_0.apply_overwrites_to_context(var_53, var_55)
    var_57 = {var_51: var_52}
    var_58 = 'not-a-boolean'
    var_59 = {var_51: var_58}
    var_60 = module_0.apply_overwrites_to_context(var_57, var_59)
    var_61 = 'nested'
    var_62 = {var_28: var_25, var_29: var_6}
    var_63 = {var_61: var_62}
    var_64 = 3
    var_65 = 4
    var_66 = {var_29: var_64, var_30: var_65}
    var_67 = {var_61: var_66}
    var_68 = module_0.apply_overwrites_to_context(var_63, var_67)



# Parsed testcases at query #2
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test render_and_create_dir with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_awesome_project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = '{{ project_name }}_dir'
    var_7 = False
    var_8 = ''
    var_9 = '{{ project_name }}_dir'
    var_10 = False
    var_11 = 'existing_dir'
    var_12 = True
    var_13 = 'prefix_{{ project_name }}_suffix'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the full generation flow:\n    1. Verifies template discovery.\n    2. Verifies directory rendering.\n    3. Verifies file rendering (Jinja2).\n    4. Verifies copy-without-render logic.\n    5. Verifies hooks are called.\n    '
    var_1 = 'repo_dir'
    var_2 = 'template_dir'
    var_3 = 'output_dir'
    var_4 = 'project_name'
    var_5 = 'cookiecutter'
    var_6 = 'MyGeneratedProject'
    var_7 = '_copy_without_render'
    var_8 = 'static_assets/*'
    var_9 = [var_8]
    var_10 = {var_7: var_9}
    var_11 = {var_4: var_6, var_5: var_10}
    var_12 = True
    var_13 = 'README.md'
    var_14 = 'static_assets'
    var_15 = 'info.txt'
    var_16 = 0

def test_case_0():
    var_0 = 'Tests that the project directory is deleted if a hook fails and keep_project_on_failure is False.'
    var_1 = 'repo_dir'
    var_2 = 'template_dir'
    var_3 = 'output_dir'
    var_4 = 'Hook Failed'
    var_5 = 'project_name'
    var_6 = 'cookiecutter'
    var_7 = 'fail_project'
    var_8 = {}
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = True
    var_11 = False



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'template.txt'
    var_1 = 'Hello {{ author }}! Welcome to {{ project_name }}.'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'binary.dat'
    var_1 = b'\x00\x01\x02\x03'
    var_2 = 'binary.dat'

def test_case_0():
    var_0 = 'exists.txt'
    var_1 = 'new content'
    var_2 = True
    var_3 = 'old content'
    assert var_3 == 'old content'

def test_case_0():
    var_0 = 'error.txt'
    var_1 = 'Hello {{ unclosed_bracket'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'template.txt'
    var_1 = 'content'
    var_2 = 'utf-8'
    var_3 = 'folder_{{ project_name }}.txt'
    var_4 = 'folder_my_project.txt'



# Parsed testcases at query #5
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = "\n    Tests the core logic of generate_files:\n    1. Template is found.\n    2. Project directory is created with rendered names.\n    3. Files are rendered correctly.\n    4. Files in '_copy_without_render' are copied without rendering.\n    5. Hooks are executed.\n    "
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'value'
    var_5 = '_copy_without_render'
    var_6 = 'test_project'
    var_7 = 'test_value'
    var_8 = 'static_dir/'
    var_9 = [var_8]
    var_10 = {var_3: var_6, var_4: var_7, var_5: var_9}
    var_11 = {var_2: var_10}
    var_12 = 'output'
    var_13 = True
    var_14 = 'config.txt'
    var_15 = 'static_dir'
    var_16 = 'data.txt'
    var_17 = -1

import jinja2.environment as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Tests that UndefinedVariableInTemplate is raised when a template variable is missing.'
    var_1 = module_0.Environment()
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'fail_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'error_output'
    var_8 = module_1.generate_files(var_0, var_6, var_2)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'old_name'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'author'
    var_6 = 'new_name'
    var_7 = 'tester'
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = module_0.apply_overwrites_to_context(var_4, var_8)
    var_10 = 'features'
    var_11 = 'logging'
    var_12 = 'testing'
    var_13 = 'auth'
    var_14 = [var_11, var_12, var_13]
    var_15 = {var_10: var_14}
    var_16 = [var_11, var_13]
    var_17 = {var_10: var_16}
    var_18 = module_0.apply_overwrites_to_context(var_15, var_17)
    var_19 = [var_11, var_12]
    var_20 = {var_10: var_19}
    var_21 = 'invalid_feature'
    var_22 = [var_21]
    var_23 = {var_10: var_22}
    var_24 = module_0.apply_overwrites_to_context(var_20, var_23)
    var_25 = 'env'
    var_26 = 'dev'
    var_27 = 'staging'
    var_28 = 'prod'
    var_29 = [var_26, var_27, var_28]
    var_30 = {var_25: var_29}
    var_31 = {var_25: var_28}
    var_32 = module_0.apply_overwrites_to_context(var_30, var_31)
    var_33 = [var_26, var_27]
    var_34 = {var_25: var_33}
    var_35 = {var_25: var_28}
    var_36 = module_0.apply_overwrites_to_context(var_34, var_35)
    var_37 = 'config'
    var_38 = 'debug'
    var_39 = 'port'
    var_40 = False
    var_41 = 8080
    var_42 = {var_38: var_40, var_39: var_41}
    var_43 = {var_37: var_42}
    var_44 = 'user'
    var_45 = True
    var_46 = 'admin'
    var_47 = {var_38: var_45, var_44: var_46}
    var_48 = {var_37: var_47}
    var_49 = module_0.apply_overwrites_to_context(var_43, var_48)
    var_50 = 'use_docker'
    var_51 = False
    var_52 = {var_50: var_51}
    var_53 = 'yes'
    var_54 = {var_50: var_53}
    var_55 = module_0.apply_overwrites_to_context(var_52, var_54)
    var_56 = 'use_docker'
    var_57 = False
    var_58 = {var_56: var_57}
    var_59 = 'not-a-boolean'
    var_60 = {var_56: var_59}
    var_61 = 'settings'
    var_62 = 'a'
    var_63 = {var_62: var_45}
    var_64 = {var_61: var_63}
    var_65 = 'b'
    var_66 = 2
    var_67 = {var_65: var_66}
    var_68 = {var_61: var_67}
    var_69 = True
    var_70 = {var_62: var_69}
    var_71 = 'new_var'
    var_72 = {var_71: var_66}
    var_73 = module_0.apply_overwrites_to_context(var_70, var_72, in_dictionary_variable=var_40)



# Parsed testcases at query #7
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.bin'
    var_3 = 'docs/manual.pdf'
    var_4 = 'static/*'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'test.bin'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True
    var_10 = module_0.is_copy_only_path(var_3, var_7)
    assert var_10 is True
    var_11 = 'static/image.png'
    var_12 = module_0.is_copy_only_path(var_11, var_7)
    assert var_12 is True
    var_13 = 'src/main.py'
    var_14 = module_0.is_copy_only_path(var_13, var_7)
    assert var_14 is False
    var_15 = 'other_key'
    var_16 = {}
    var_17 = {var_15: var_16}
    var_18 = module_0.is_copy_only_path(var_8, var_17)
    assert var_18 is False
    var_19 = {}
    var_20 = {var_0: var_19}
    var_21 = []
    var_22 = {var_1: var_21}
    var_23 = {var_0: var_22}
    var_24 = module_0.is_copy_only_path(var_8, var_23)
    assert var_24 is False
    var_25 = 'exact_match.txt'
    var_26 = [var_25]
    var_27 = {var_1: var_26}
    var_28 = {var_0: var_27}
    var_29 = module_0.is_copy_only_path(var_25, var_28)
    assert var_29 is True



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'hello_{{ name }}.txt'
    var_1 = 'Hello, {{ name }}!'
    var_2 = 'utf-8'
    var_3 = 'name'
    var_4 = 'cookiecutter'
    var_5 = 'World'
    var_6 = '_new_lines'
    var_7 = '\n'
    var_8 = {var_6: var_7}
    var_9 = {var_3: var_5, var_4: var_8}
    var_10 = 'hello_World.txt'
    var_11 = 'Hello, World!\n'
    var_12 = {var_10: var_11}
    var_13 = 'hello_{{ name }}.txt'
    var_14 = 'simple.txt'
    var_15 = 'Content: {{ var }}'
    var_16 = 'var'
    var_17 = 'test'
    var_18 = {var_6: var_7}
    var_19 = {var_16: var_17, var_4: var_18}

def test_case_0():
    var_0 = 'exists.txt'
    var_1 = 'Original'
    var_2 = 'utf-8'
    var_3 = "Don't overwrite me"
    var_4 = 'cookiecutter'
    var_5 = {}
    var_6 = {var_4: var_5}
    var_7 = True

def test_case_0():
    var_0 = 'template_file.txt'
    var_1 = 'Inside'
    var_2 = 'utf-str'
    var_3 = 'sub'
    var_4 = 'file.txt'
    var_5 = 'Hello'
    var_6 = 'utf-8'
    var_7 = 'name'
    var_8 = 'cookiecutter'
    var_9 = 'Project'
    var_10 = '_new_lines'
    var_11 = '\n'
    var_12 = {var_10: var_11}
    var_13 = {var_7: var_9, var_8: var_12}
    var_14 = 'sub/file.txt'
    var_15 = 'sub/file.txt'



# Parsed testcases at query #9
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'Test the render_and_create_dir function with various scenarios.'
    var_1 = module_0.Environment()
    var_2 = 'project_name'
    var_3 = 'my_project'
    var_4 = {var_2: var_3}
    var_5 = 'output'
    var_6 = '{{ project_name }}_dir'
    var_7 = False
    var_8 = ''
    var_9 = False
    var_10 = 'old_file.txt'
    var_11 = 'old content'
    var_12 = '{{ project_name }}_dir'
    var_13 = True



# Parsed testcases at query #10
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Tests the generate_context function for various scenarios.'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'version'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = '0.1.0'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'overridden_name'
    var_11 = {var_2: var_10}
    var_12 = 'default_name'
    var_13 = {var_2: var_12}
    var_14 = 'bad.json'
    var_15 = "{ 'invalid': json }"
    var_16 = module_1.generate_context(var_0)
    var_17 = 'non_existent_file.json'
    var_18 = module_1.generate_context(var_17)
    var_19 = 'complex.json'
    var_20 = 'settings'
    var_21 = 'debug'
    var_22 = 'features'
    var_23 = False
    var_24 = 'auth'
    var_25 = 'api'
    var_26 = [var_24, var_25]
    var_27 = {var_21: var_23, var_22: var_26}
    var_28 = {var_20: var_27}
    var_29 = 'complex'
    var_30 = {var_29: var_28}
    var_31 = module_0.dumps(var_30)
    var_32 = 'new_key'
    var_33 = True
    var_34 = 'new_val'
    var_35 = {var_21: var_33, var_32: var_34}
    var_36 = {var_20: var_35}



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = '\n    Tests the main entry point for generating a project.\n    Verifies that files are rendered, directories are created, \n    and hooks are called.\n    '
    var_1 = 'template_dir'
    var_2 = 'output_dir'
    var_3 = 'context'
    var_4 = 'project_name'
    var_5 = 'my_awesome_project'
    var_6 = 'rendered_content'
    var_7 = True
    var_8 = 'README.md'
    var_9 = 'src'
    var_10 = 'main.py'
    var_11 = 'static'
    var_12 = 'logo.png'

import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'Tests that UndefinedVariableInTemplate is raised when a template variable is missing.'
    var_1 = 'template_dir'
    var_2 = 'broken.txt'
    var_3 = '{{ non_existent_variable }}'
    var_4 = 'broken_project'
    var_5 = 'cookiecutter.json'
    var_6 = 'project_name'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = {var_6: var_7}
    var_11 = 'output_dir'
    var_12 = module_1.generate_files(var_0, var_10, var_2)



