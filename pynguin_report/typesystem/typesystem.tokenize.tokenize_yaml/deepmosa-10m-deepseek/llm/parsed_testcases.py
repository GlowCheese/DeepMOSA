####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 12/13 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 11/12 statements.


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = '   \n  \t  '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.base as module_1


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b\n- c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b', 'c'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b\n- c'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 9
    var_12 = module_1.Position(var_10, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 11
    var_12 = 13
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'key: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'
    var_7 = 1
    var_8 = 6
    var_9 = 5
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = 10
    var_14 = 9
    var_15 = module_1.Position(var_7, var_13, var_14)
    var_16 = var_4.end
    var_17 = bool(var_4.end == var_15)
    assert var_17 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'
    var_7 = 1
    var_8 = 0
    var_9 = module_1.Position(var_7, var_7, var_8)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 3
    var_13 = 2
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.tokenize.tokenize_yaml as module_2


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.Field()
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True


def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.Field()
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)


def test_case_0():
    var_0 = ''
    var_1 = module_0.Field()
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)


def test_case_0():
    var_0 = b'key: value'
    var_1 = module_0.Field()
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)
    var_7 = bool(var_6 == {'key': 'value'})
    assert var_7 is True


def test_case_0():
    var_0 = 'key: 123'
    var_1 = module_0.Field()
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)
    var_7 = bool(var_6 == {'key': 123})
    assert var_7 is True


def test_case_0():
    var_0 = '{}'
    var_1 = module_0.Field()
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.validate_yaml(var_0, var_5)


def test_case_0():
    var_0 = '{}'
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1)
    var_3 = 'key'
    var_4 = {var_3: var_2}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(var_7 == {'key': 'default_value'})
    assert var_8 is True


def test_case_0():
    var_0 = 'key: null'
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = 'key'
    var_4 = {var_3: var_2}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(var_7 == {'key': None})
    assert var_8 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = 'key'
    var_4 = {var_3: var_2}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True


def test_case_0():
    var_0 = 'outer:\n  inner: value'
    var_1 = module_0.Field()
    var_2 = 'inner'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'outer'
    var_7 = {var_6: var_5}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(var_10 == {'outer': {'inner': 'value'}})
    assert var_11 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_tokenize_yaml_parse_error_without_problem_mark. Retrieved 5/19 statements.


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = 'test problem'
    var_1 = None
    var_2 = []
    var_3 = 'some content'
    var_4 = module_0.tokenize_yaml(var_3)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_tokenize_yaml_parse_error_without_problem. Retrieved 2/12 statements.


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = 'invalid: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #6
#--------------------------





def test_case_0():
    var_0 = 'invalid: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_multiline_scalar. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_anchors_and_aliases. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_complex_mapping. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_quoted_scalars. Retrieved 8/9 statements.



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = '   \n  \t  '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.base as module_1


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 12
    var_12 = 14
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = '|\n  line1\n  line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'line1\nline2\n'
    var_3 = var_1.string
    assert var_3 == '|\n  line1\n  line2'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 7
    var_11 = 19
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = '&anchor value\n*anchor'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['value', 'value'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '&anchor value\n*anchor'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 7
    var_12 = 20
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = '? complex key\n: complex value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'complex key': 'complex value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '? complex key\n: complex value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 15
    var_12 = 30
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = '\'single\' "double"'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['single', 'double'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '\'single\' "double"'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 17
    var_11 = 16
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_yaml_does_not_raise_assertion_error_when_yaml_is_installed. Retrieved 3/9 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.Field()
    var_2 = module_1.validate_yaml(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_tokenize_yaml_assertion_error_when_yaml_is_none. Retrieved 3/9 statements.


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = 'yaml'
    var_1 = 'test'
    var_2 = module_0.tokenize_yaml(var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = '   \n\t  '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.base as module_1


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b\n- c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b', 'c'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b\n- c'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 10
    var_12 = module_1.Position(var_10, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 11
    var_12 = 14
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = 'a: [x: 1]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = 'x'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'name: John'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = bool(var_7 == {'name': 'John'})
    assert var_8 is True


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'name: John\n  invalid: indent'
    var_7 = module_2.validate_yaml(var_6, var_5)


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = ''
    var_7 = module_2.validate_yaml(var_6, var_5)


def test_case_0():
    var_0 = 'name'
    var_1 = 5
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, **var_2)
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'name: Jonathan'
    var_8 = module_2.validate_yaml(var_7, var_6)
    var_9 = len(e.messages())
    assert var_9 == 1
    var_10 = e.messages()[0]
    var_11 = var_10.code
    assert var_11 == 'max_length'
    var_12 = var_10.start_position
    var_13 = bool(var_10.start_position is not None)
    assert var_13 is True


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'age: 30'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0]
    var_10 = var_9.code
    assert var_10 == 'required'
    var_11 = var_9.start_position
    var_12 = bool(var_9.start_position is not None)
    assert var_12 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'Unknown'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = '{}'
    var_9 = module_2.validate_yaml(var_8, var_7)
    var_10 = bool(var_9 == {'name': 'Unknown'})
    assert var_10 is True


def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'name: null'
    var_9 = module_2.validate_yaml(var_8, var_7)
    var_10 = bool(var_9 == {'name': None})
    assert var_10 is True


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = b'name: Alice'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = bool(var_7 == {'name': 'Alice'})
    assert var_8 is True


def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'person'
    var_7 = {var_6: var_5}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = 'person:\n  age: 25'
    var_11 = module_2.validate_yaml(var_10, var_9)
    var_12 = bool(var_11 == {'person': {'age': 25}})
    assert var_12 is True


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = '123: value'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = len(e.messages())
    assert var_8 == 1
    var_9 = e.messages()[0]
    var_10 = var_9.code
    assert var_10 == 'invalid_key'



# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_multiline_scalar. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.base as module_1


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': [1, 2]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key:\n  - 1\n  - 2'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 5
    var_12 = 16
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'


def test_case_0():
    var_0 = '|\n  line1\n  line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'line1\nline2\n'
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 3
    var_9 = 6
    var_10 = 19
    var_11 = module_1.Position(var_8, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '   \n   '
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_tokenize_yaml_parse_error_without_problem. Retrieved 4/8 statements.


import yaml.scanner as module_0


def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = module_0.ScannerError(var_0, var_1, var_1, var_1, var_1)
    var_3 = ''



# Parsed testcases at query #16
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = 'invalid: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenize_yaml_parse_error_without_problem. Retrieved 4/8 statements.


import yaml.scanner as module_0


def test_case_0():
    var_0 = ''
    var_1 = None
    var_2 = module_0.ScannerError(var_0, var_1, var_1, var_1, var_1)
    var_3 = 'invalid yaml'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_tokenize_yaml_assertion_error_when_yaml_is_none. Retrieved 2/8 statements.


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = 'test'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #20
#--------------------------





def test_case_0():
    var_0 = 'invalid: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_anchors_and_aliases. Retrieved 2/3 statements.



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = '   \n  \t  '
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 11
    var_12 = 13
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'


def test_case_0():
    var_0 = 'key: |\n  line1\n  line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'line1\nline2\n'})
    assert var_3 is True
    var_4 = var_1.string
    var_5 = bool(var_1.string == var_0)
    assert var_5 is True


def test_case_0():
    var_0 = '&anchor value\nother: *anchor'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'other': 'value'})
    assert var_3 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_tokenize_yaml_parse_error_without_problem. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'MockError'
    var_1 = ()
    var_2 = 'problem'
    var_3 = 'problem_mark'
    var_4 = None
    var_5 = {var_2: var_4, var_3: var_4}
    var_6 = [var_0, var_1, var_5]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_yaml_raises_assertion_error_when_yaml_is_none. Retrieved 3/9 statements.



def test_case_0():
    var_0 = 'test: value'
    var_1 = None
    var_2 = module_0.validate_yaml(var_0, var_1)



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1


def test_case_0():
    var_0 = None
    var_1 = module_0.Field()
    var_2 = 'test: value'
    var_3 = module_1.validate_yaml(var_2, var_1)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = 'invalid: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #28
#--------------------------





def test_case_0():
    var_0 = 'invalid yaml content: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #29
#--------------------------





def test_case_0():
    var_0 = 'invalid: yaml: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup_dict. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_list. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 9/10 statements.



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = '   \n  \t  '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.base as module_1


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 10
    var_12 = 13
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'could not find expected key'


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'a'
    var_6 = var_4.string
    assert var_6 == 'a'


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'


def test_case_0():
    var_0 = '|\n  line1\n  line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'line1\nline2\n'
    var_3 = var_1.string
    assert var_3 == '|\n  line1\n  line2'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 6
    var_11 = 18
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True



# Parsed testcases at query #31
#--------------------------






# Parsed testcases at query #32
#--------------------------

# Partially parsed test_tokenize_yaml_parse_error_without_problem_mark. Retrieved 3/7 statements.


import yaml.parser as module_0


def test_case_0():
    var_0 = 'problem'
    var_1 = module_0.ParserError(var_0)
    var_2 = 'invalid: ['



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup_dict. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_list. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_multiline_scalar. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_anchors_and_aliases. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_complex_mapping. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = '   \n  \t  '
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b\n- c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b', 'c'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b\n- c'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 9
    var_12 = module_1.Position(var_10, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 11
    var_12 = 13
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'b'
    var_6 = var_4.string
    assert var_6 == 'b'


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'


def test_case_0():
    var_0 = '|\n  line1\n  line2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'line1\nline2\n'
    var_3 = 1
    var_4 = 0
    var_5 = module_1.Position(var_3, var_3, var_4)
    var_6 = var_1.start
    var_7 = bool(var_1.start == var_5)
    assert var_7 is True
    var_8 = 3
    var_9 = 6
    var_10 = 18
    var_11 = module_1.Position(var_8, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '&anchor value\nother: *anchor'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'other': 'value'})
    assert var_3 is True
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = 14
    var_11 = 27
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'a: 1\nb:\n  c: 2\n  d: 3'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': 1, 'b': {'c': 2, 'd': 3}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a: 1\nb:\n  c: 2\n  d: 3'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 6
    var_12 = 22
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = module_0.tokenize_yaml(var_0)
    var_3 = bool(var_1 == var_2)
    assert var_3 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key: other'
    var_3 = module_0.tokenize_yaml(var_2)
    var_4 = bool(not var_1 == var_3)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_tokenize_yaml_assertion_error_when_yaml_not_installed. Retrieved 2/8 statements.



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': [1, 2]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key:\n  - 1\n  - 2'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 5
    var_12 = 16
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'did not find expected node content'



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name: John\nage: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(var_10 == {'name': 'John', 'age': '30'})
    assert var_11 is True


def test_case_0():
    var_0 = 'name: John\n  age: 30'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(False)
    assert var_11 is True


def test_case_0():
    var_0 = ''
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(False)
    assert var_8 is True


def test_case_0():
    var_0 = 'name: John'
    var_1 = 'name'
    var_2 = 3
    var_3 = {}
    var_4 = module_0.String(max_length=var_2, **var_3)
    var_5 = {var_1: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = module_2.validate_yaml(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = len(exc.messages())
    assert var_10 == 1
    var_11 = exc.messages()[0]
    var_12 = var_11.start_position
    var_13 = bool(var_11.start_position is not None)
    assert var_13 is True
    var_14 = var_11.end_position
    var_15 = bool(var_11.end_position is not None)
    assert var_15 is True


def test_case_0():
    var_0 = '{}'
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(exc.messages())
    assert var_9 == 1
    var_10 = exc.messages()[0]
    var_11 = var_10.code
    assert var_11 == 'required'
    var_12 = var_10.index
    var_13 = bool(var_10.index == ['name'])
    assert var_13 is True


def test_case_0():
    var_0 = '{}'
    var_1 = 'name'
    var_2 = 'default_name'
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = module_2.validate_yaml(var_0, var_8)
    var_10 = bool(var_9 == {'name': 'default_name'})
    assert var_10 is True


def test_case_0():
    var_0 = 'name: null'
    var_1 = 'name'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {var_1: var_5}
    var_7 = 'allow_null'
    var_8 = {var_7: var_2}
    var_9 = module_1.Schema(var_6, **var_8)
    var_10 = module_2.validate_yaml(var_0, var_9)
    var_11 = bool(var_10 == {'name': None})
    assert var_11 is True


def test_case_0():
    var_0 = b'name: Alice'
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(var_7 == {'name': 'Alice'})
    assert var_8 is True


def test_case_0():
    var_0 = 'person:\n  name: Bob\n  age: 25'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = 'person'
    var_11 = {var_10: var_9}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = module_2.validate_yaml(var_0, var_13)
    var_15 = bool(var_14 == {'person': {'name': 'Bob', 'age': 25}})
    assert var_15 is True


def test_case_0():
    var_0 = '123: value'
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = module_2.validate_yaml(var_0, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(exc.messages())
    var_10 = bool(len(exc.messages()) >= 1)
    assert var_10 is True
    var_11 = exc.messages()[0]
    var_12 = var_11.code
    assert var_12 == 'invalid_key'



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_yaml_does_not_raise_assertion_error_when_yaml_is_installed. Retrieved 3/10 statements.


import typesystem.tokenize.tokenize_yaml as module_1


def test_case_0():
    var_0 = []
    var_1 = 'key: value'
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_yaml_raises_assertion_error_when_yaml_is_none. Retrieved 4/12 statements.



def test_case_0():
    var_0 = 'yaml'
    var_1 = ''
    var_2 = module_0.Field()
    var_3 = module_1.validate_yaml(var_1, var_2)



# Parsed testcases at query #10
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = 'invalid yaml: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #11
#--------------------------





def test_case_0():
    var_0 = 'invalid: yaml: :'
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_1
import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = ''
    var_3 = module_0.Token(var_0, var_1, var_1, var_2)
    var_4 = 'key: value'
    var_5 = module_1.Field()
    var_6 = module_2.validate_yaml(var_4, var_5)
    assert var_6 == 42



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 12/13 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 11/12 statements.


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = '   \n   '
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.base as module_1


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b\n- c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b', 'c'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b\n- c'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 8
    var_12 = module_1.Position(var_10, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 11
    var_12 = 13
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'could not find expected key'


def test_case_0():
    var_0 = 'a:\n  - x: 1\n    y: 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'a'
    var_3 = 0
    var_4 = 'x'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.lookup(var_5)
    var_7 = var_6.value
    assert var_7 == 1
    var_8 = var_6.string
    assert var_8 == '1'
    var_9 = 2
    var_10 = 8
    var_11 = 12
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_6.start
    var_14 = bool(var_6.start == var_12)
    assert var_14 is True
    var_15 = module_1.Position(var_9, var_10, var_11)
    var_16 = var_6.end
    var_17 = bool(var_6.end == var_15)
    assert var_17 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'
    var_7 = 1
    var_8 = 0
    var_9 = module_1.Position(var_7, var_7, var_8)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 3
    var_13 = 2
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_tokenize_yaml_parse_error_without_problem. Retrieved 5/9 statements.


import yaml.scanner as module_0


def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = module_0.ScannerError(var_0, var_1, var_1, var_0, var_1)
    var_3 = 0
    var_4 = None



# Parsed testcases at query #15
#--------------------------




import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = 'invalid: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #16
#--------------------------





def test_case_0():
    var_0 = 'invalid: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #17
#--------------------------





def test_case_0():
    var_0 = 'invalid: ['
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_tokenize_yaml_parse_error_without_problem_mark. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'invalid: ['



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_anchors_and_aliases. Retrieved 9/10 statements.



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'key:\n  - 1\n  - 2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': [1, 2]})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key:\n  - 1\n  - 2'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 5
    var_12 = 15
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'did not find expected node content'


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'


def test_case_0():
    var_0 = '|\n  hello\n  world'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello\nworld\n'
    var_3 = var_1.string
    assert var_3 == '|\n  hello\n  world'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 7
    var_11 = 20
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = '&anchor value\nother: *anchor'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'other': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '&anchor value\nother: *anchor'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 14
    var_12 = 27
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 2/3 statements.
# Partially parsed test_tokenize_yaml_token_lookup. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_token_lookup_key. Retrieved 5/6 statements.



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = b'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'


def test_case_0():
    var_0 = 'key:\n  - 1\n  - two'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': [1, 'two']})
    assert var_3 is True


def test_case_0():
    var_0 = 'key: ['
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_1.Position(var_2, var_2, var_3)
    var_5 = var_1.start
    var_6 = bool(var_1.start == var_4)
    assert var_6 is True
    var_7 = 11
    var_8 = 10
    var_9 = module_1.Position(var_2, var_7, var_8)
    var_10 = var_1.end
    var_11 = bool(var_1.end == var_9)
    assert var_11 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_yaml_validation_error_with_positions. Retrieved 6/9 statements.
# Partially parsed test_validate_yaml_required_field_missing. Retrieved 6/9 statements.
# Partially parsed test_validate_yaml_with_union_field_error. Retrieved 6/8 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'age: 25'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = bool(var_7 == {'age': 25})
    assert var_8 is True


def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'age: :'
    var_7 = module_2.validate_yaml(var_6, var_5)


def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = ''
    var_7 = module_2.validate_yaml(var_6, var_5)


def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'age: invalid'
    var_7 = module_2.validate_yaml(var_6, var_5)


def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'name: John'
    var_7 = module_2.validate_yaml(var_6, var_5)


def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = b'age: 30'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = bool(var_7 == {'age': 30})
    assert var_8 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'name: Alice\nage: 28'
    var_10 = module_2.validate_yaml(var_9, var_8)
    var_11 = bool(var_10 == {'name': 'Alice', 'age': 28})
    assert var_11 is True

import typesystem.tokenize.tokenize_yaml as module_1


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = '42'
    var_8 = module_1.validate_yaml(var_7, var_6)
    assert var_8 == 42


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = '"hello"'
    var_8 = module_1.validate_yaml(var_7, var_6)
    assert var_8 == 'hello'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'null'
    var_8 = module_1.validate_yaml(var_7, var_6)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_tokenize_yaml_simple_scalar. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_integer. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_boolean_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_boolean_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup. Retrieved 12/13 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 11/12 statements.


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = '   \n\t  '
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True

import typesystem.base as module_1


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b\n- c'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b', 'c'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b\n- c'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 8
    var_12 = module_1.Position(var_10, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'list:\n  - item1\n  - item2'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'list': ['item1', 'item2']})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'list:\n  - item1\n  - item2'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 3
    var_11 = 8
    var_12 = 24
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'key: [unclosed'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'
    var_7 = 1
    var_8 = 6
    var_9 = 5
    var_10 = module_1.Position(var_7, var_8, var_9)
    var_11 = var_4.start
    var_12 = bool(var_4.start == var_10)
    assert var_12 is True
    var_13 = 10
    var_14 = 9
    var_15 = module_1.Position(var_7, var_13, var_14)
    var_16 = var_4.end
    var_17 = bool(var_4.end == var_15)
    assert var_17 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'
    var_7 = 1
    var_8 = 0
    var_9 = module_1.Position(var_7, var_7, var_8)
    var_10 = var_4.start
    var_11 = bool(var_4.start == var_9)
    assert var_11 is True
    var_12 = 3
    var_13 = 2
    var_14 = module_1.Position(var_7, var_12, var_13)
    var_15 = var_4.end
    var_16 = bool(var_4.end == var_14)
    assert var_16 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_yaml_positional_error_messages. Retrieved 8/12 statements.


import typesystem.fields as module_0
import typesystem.tokenize.tokenize_yaml as module_1


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = '42'
    var_3 = module_1.validate_yaml(var_2, var_1)
    assert var_3 == 42


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = '['
    var_3 = module_1.validate_yaml(var_2, var_1)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = ''
    var_3 = module_1.validate_yaml(var_2, var_1)

import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'name: John'
    var_7 = module_2.validate_yaml(var_6, var_5)
    var_8 = bool(var_7 == {'name': 'John'})
    assert var_8 is True


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'age: 30'
    var_7 = module_2.validate_yaml(var_6, var_5)

import typesystem.tokenize.tokenize_yaml as module_1


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = '123'
    var_6 = module_1.validate_yaml(var_5, var_4)
    assert var_6 == 123


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = 'true'
    var_6 = module_1.validate_yaml(var_5, var_4)


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = b'hello'
    var_3 = module_1.validate_yaml(var_2, var_1)
    assert var_3 == 'hello'

import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = '{123: value}'
    var_7 = module_2.validate_yaml(var_6, var_5)


def test_case_0():
    var_0 = 'name'
    var_1 = 'Anonymous'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = '{}'
    var_9 = module_2.validate_yaml(var_8, var_7)
    var_10 = bool(var_9 == {'name': 'Anonymous'})
    assert var_10 is True


def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = 'read_only'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'name: John'
    var_9 = module_2.validate_yaml(var_8, var_7)
    var_10 = bool(var_9 == {})
    assert var_10 is True


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_1.Schema(var_3, **var_6)
    var_8 = 'null'
    var_9 = module_2.validate_yaml(var_8, var_7)
    assert var_9 is None


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_1.Schema(var_3, **var_6)
    var_8 = 'null'
    var_9 = module_2.validate_yaml(var_8, var_7)


def test_case_0():
    var_0 = 'person'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = {var_0: var_9}
    var_11 = {}
    var_12 = module_1.Schema(var_10, **var_11)
    var_13 = 'person:\n  name: Alice\n  age: 25'
    var_14 = module_2.validate_yaml(var_13, var_12)
    var_15 = bool(var_14 == {'person': {'name': 'Alice', 'age': 25}})
    assert var_15 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'name: John\nage: not_a_number'
    var_10 = module_2.validate_yaml(var_9, var_8)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_lookup_list. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_dict. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_lookup_key. Retrieved 5/6 statements.
# Partially parsed test_tokenize_yaml_multiline_string. Retrieved 9/10 statements.


import typesystem.tokenize.tokenize_yaml as module_0


def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = b''
    var_1 = module_0.tokenize_yaml(var_0)

import typesystem.base as module_1


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 10
    var_12 = 13
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 0
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'a'
    var_6 = var_4.string
    assert var_6 == 'a'


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup(var_3)
    var_5 = var_4.value
    assert var_5 == 'value'
    var_6 = var_4.string
    assert var_6 == 'value'


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = var_1.lookup_key(var_3)
    var_5 = var_4.value
    assert var_5 == 'key'
    var_6 = var_4.string
    assert var_6 == 'key'


def test_case_0():
    var_0 = '|\n  hello\n  world'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello\nworld\n'
    var_3 = var_1.string
    assert var_3 == '|\n  hello\n  world'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 3
    var_10 = 7
    var_11 = 20
    var_12 = module_1.Position(var_9, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True



# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------





def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_tokenize_yaml_asserts_yaml_is_not_none. Retrieved 4/11 statements.



def test_case_0():
    var_0 = 'typesystem.tokenize.tokenize_yaml'
    var_1 = ''
    var_2 = module_0.tokenize_yaml(var_1)
    var_3 = 'yaml'



# Parsed testcases at query #29
#--------------------------






# Parsed testcases at query #30
#--------------------------

# Partially parsed test_tokenize_yaml_scalar_string. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_int. Retrieved 7/8 statements.
# Partially parsed test_tokenize_yaml_scalar_float. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_true. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_bool_false. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_scalar_null. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_list. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_dict. Retrieved 8/9 statements.
# Partially parsed test_tokenize_yaml_nested_structure. Retrieved 9/10 statements.
# Partially parsed test_tokenize_yaml_bytes_input. Retrieved 8/9 statements.



def test_case_0():
    var_0 = ''
    var_1 = module_0.tokenize_yaml(var_0)


def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '42'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 42
    var_3 = var_1.string
    assert var_3 == '42'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 2
    var_10 = module_1.Position(var_4, var_9, var_4)
    var_11 = var_1.end
    var_12 = bool(var_1.end == var_10)
    assert var_12 is True


def test_case_0():
    var_0 = '3.14'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == 3.14)
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '3.14'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 4
    var_11 = 3
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'true'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is True
    var_3 = var_1.string
    assert var_3 == 'true'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'false'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is False
    var_3 = var_1.string
    assert var_3 == 'false'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = 'null'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 is None
    var_3 = var_1.string
    assert var_3 == 'null'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 4
    var_10 = 3
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = '- a\n- b'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == ['a', 'b'])
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == '- a\n- b'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 3
    var_12 = 6
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = 'key: value'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'key': 'value'})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'key: value'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 10
    var_11 = 9
    var_12 = module_1.Position(var_5, var_10, var_11)
    var_13 = var_1.end
    var_14 = bool(var_1.end == var_12)
    assert var_14 is True


def test_case_0():
    var_0 = 'a:\n  b: [1, 2]'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    var_3 = bool(var_1.value == {'a': {'b': [1, 2]}})
    assert var_3 is True
    var_4 = var_1.string
    assert var_4 == 'a:\n  b: [1, 2]'
    var_5 = 1
    var_6 = 0
    var_7 = module_1.Position(var_5, var_5, var_6)
    var_8 = var_1.start
    var_9 = bool(var_1.start == var_7)
    assert var_9 is True
    var_10 = 2
    var_11 = 11
    var_12 = 13
    var_13 = module_1.Position(var_10, var_11, var_12)
    var_14 = var_1.end
    var_15 = bool(var_1.end == var_13)
    assert var_15 is True


def test_case_0():
    var_0 = b'hello'
    var_1 = module_0.tokenize_yaml(var_0)
    var_2 = var_1.value
    assert var_2 == 'hello'
    var_3 = var_1.string
    assert var_3 == 'hello'
    var_4 = 1
    var_5 = 0
    var_6 = module_1.Position(var_4, var_4, var_5)
    var_7 = var_1.start
    var_8 = bool(var_1.start == var_6)
    assert var_8 is True
    var_9 = 5
    var_10 = 4
    var_11 = module_1.Position(var_4, var_9, var_10)
    var_12 = var_1.end
    var_13 = bool(var_1.end == var_11)
    assert var_13 is True


def test_case_0():
    var_0 = ': invalid'
    var_1 = module_0.tokenize_yaml(var_0)



