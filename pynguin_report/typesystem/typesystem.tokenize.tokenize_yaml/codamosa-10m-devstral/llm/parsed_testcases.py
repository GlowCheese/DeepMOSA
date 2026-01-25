####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name: 123'
    var_2 = 1
    var_3 = 'items:\n  - 1\n  - 2\n  - 3'
    var_4 = 'name: John\ninvalid: yaml: content'
    var_5 = ''
    var_6 = 'user:\n  name: John\n  age: 30'
    var_7 = b'name: John'



# Parsed testcases at query #2
#--------------------------


import typesystem.tokenize.tokenize_yaml as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'hello'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = 'key: value'
    var_5 = module_0.tokenize_yaml(var_4)
    var_6 = '- item1\n- item2'
    var_7 = module_0.tokenize_yaml(var_6)
    var_8 = 'int: 1\nfloat: 1.5\nbool: true\nnull: null'
    var_9 = module_0.tokenize_yaml(var_8)
    var_10 = 'invalid: yaml: content'
    var_11 = module_0.tokenize_yaml(var_10)
    var_12 = b'key: value'
    var_13 = module_0.tokenize_yaml(var_12)
    var_14 = 'list:\n  - item1\n  - item2\nnested:\n  key: value'
    var_15 = module_0.tokenize_yaml(var_14)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '\nname: John\nage: 30\n'
    var_1 = '\nname: John\nage: thirty\n'
    var_2 = '\nname: John\nage: -5\n'
    var_3 = 'age'
    var_4 = 'field'
    var_5 = ''
    var_6 = '\nname: John\n'
    var_7 = '\nname: John\nage: 30\nextra_field: value\n'
    var_8 = '\nuser:\n  name: John\n  age: 30\nsettings:\n  - dark_mode: true\n  - notifications: false\n'
    var_9 = b'\nname: John\nage: 30\n'



