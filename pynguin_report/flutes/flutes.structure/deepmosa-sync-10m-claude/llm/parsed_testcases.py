####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 7/10 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dict. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_mixed_nested_structure. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_scalar_values. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_empty_list. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_preserves_tuple_structure. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = (var_0, var_1)
    var_3 = 'c'
    var_4 = 'd'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 5
    var_8 = {var_2: var_7}
    var_9 = 4
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 10
    var_8 = 20
    var_9 = [var_7, var_8]
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = [var_6, var_11]

def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = {var_4, var_5, var_6}
    var_8 = [var_3, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'Structures cannot contain `set`'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 10/13 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_preserves_ordered_dict. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 2
    var_6 = 3
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_0: var_2, var_1: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = 4
    var_7 = 5
    var_8 = (var_6, var_7)
    var_9 = [var_0, var_5, var_8]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_tuple. Retrieved 1/4 statements.
# Partially parsed test_map_structure_preserves_namedtuple_type. Retrieved 6/13 statements.
# Partially parsed test_map_structure_with_nested_namedtuple. Retrieved 9/20 statements.
# Partially parsed test_map_structure_with_string_function. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 9/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'data'
    var_1 = 'value'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = {var_0: var_6, var_1: var_7}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_line_15_evaluates_to_false. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]
    var_5 = 0
    var_6 = var_4[var_5]
    var_7 = '__no_map__'
    var_8 = var_6.__class__
    var_9 = hasattr(var_6, var_7)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/18 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_three_arguments. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.
# Partially parsed test_map_structure_zip_with_empty_list. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_empty_dict. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = [var_1, var_3]
    var_8 = 5
    var_9 = [var_4, var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = 3
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 4
    var_6 = 5
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = {var_0: var_6}
    var_8 = 4
    var_9 = {var_0: var_8}
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'set'
    var_9 = bool('set' in str(e).lower())
    assert var_9 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = [var_0, var_1]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_map_structure_predicate_line_1. Retrieved 5/27 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = ()
    var_5 = '__no_map__'
    var_6 = '__wrapped__'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/18 statements.
# Partially parsed test_map_structure_zip_with_scalar_values. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_mixed_nested_structure. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_complex_function. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples_and_lists. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = (var_0, var_1, var_2)
    var_4 = 5
    var_5 = 6
    var_6 = 7
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = [var_6, var_7]
    var_9 = 6
    var_10 = [var_9]
    var_11 = [var_8, var_10]
    var_12 = [var_5, var_11]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 5
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'list'
    var_1 = 'val'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 10
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 3
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = 20
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = [var_6, var_11]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = [var_1, var_3]
    var_8 = 5
    var_9 = [var_4, var_8]
    var_10 = (var_7, var_9)
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Structures cannot contain `set`'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 11/14 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_string_function. Retrieved 3/6 statements.
# Partially parsed test_map_structure_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_empty_tuple. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 'c'
    var_7 = 4
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = {var_0: var_4, var_1: var_9}

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 3
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_preserves_dict_key_order. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = [var_1, var_3]
    var_8 = 5
    var_9 = [var_4, var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = 3
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 4
    var_6 = 5
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = {var_0: var_6}
    var_8 = 4
    var_9 = {var_0: var_8}
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'set'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 'z'
    var_1 = 'a'
    var_2 = 'm'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 4
    var_8 = 5
    var_9 = 6
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = [var_6, var_10]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_map_structure_with_simple_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_tuple. Retrieved 1/4 statements.
# Partially parsed test_map_structure_preserves_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 13/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'dict'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 'x'
    var_10 = 5
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = 4
    var_5 = [var_2, var_0, var_3, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [1, 4, 9, 16])
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 6/32 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__wrapped__'
    var_5 = None
    var_6 = '__no_type_check__'
    var_7 = False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1. Retrieved 4/28 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = '__call__'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_15_evaluates_to_false. Retrieved 9/30 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = var_8.__class__
    var_10 = hasattr(var_8, var_4)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/9 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_preserves_list_type. Retrieved 4/8 statements.
# Partially parsed test_map_structure_preserves_tuple_type. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_function_returning_string. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_tuple. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 5/25 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = ()
    var_5 = '__no_map__'
    var_6 = None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Failed to parse test_map_structure_decorator_exists.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 6/27 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__wrapped__'
    var_5 = None
    var_6 = False
    assert var_6 is False
    var_7 = '@no_type_check'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_map_structure_decorator_no_type_check. Retrieved 3/9 statements.


def test_case_0():
    var_0 = '__wrapped__'
    var_1 = '__no_type_check__'
    var_2 = False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_15_evaluates_to_false. Retrieved 10/31 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_8]
    var_10 = var_8.__class__
    var_11 = hasattr(var_8, var_4)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_evaluates_to_false. Retrieved 9/30 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = var_8.__class__
    var_10 = hasattr(var_8, var_4)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_15_false. Retrieved 11/32 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_8]
    var_10 = (var_9,)
    var_11 = var_8.__class__
    var_12 = hasattr(var_8, var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_nested_mixed_structures. Retrieved 9/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 24/32 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_7]
    var_9 = lambda x, y: x + y
    var_10 = [var_4, var_5]
    var_11 = 4
    var_12 = [var_6, var_11]
    var_13 = [var_10, var_12]
    var_14 = module_0.map_structure_zip(var_9, var_13)
    var_15 = bool(var_14 == [4, 6])
    assert var_15 is True
    var_16 = lambda x: x * var_5
    var_17 = 'a'
    var_18 = 'b'
    var_19 = {var_17: var_4, var_18: var_5}
    var_20 = [var_19]
    var_21 = module_0.map_structure_zip(var_16, var_20)
    var_22 = bool(var_21 == {'a': 2, 'b': 4})
    assert var_22 is True
    var_23 = lambda x: x * var_5
    var_24 = (var_4, var_5, var_6)
    var_25 = [var_24]
    var_26 = module_0.map_structure_zip(var_23, var_25)
    var_27 = bool(var_26 == (2, 4, 6))
    assert var_27 is True
    var_28 = '__name__'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_map_structure_zip_predicate. Retrieved 12/31 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = lambda x, y: x + y
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = 3
    var_10 = 4
    var_11 = [var_9, var_10]
    var_12 = [var_8, var_11]
    var_13 = module_0.map_structure_zip(var_5, var_12)
    var_14 = bool(var_13 == [4, 6])
    assert var_14 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_scalar. Retrieved 3/9 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.
# Partially parsed test_map_structure_zip_with_complex_nested_structure. Retrieved 18/21 statements.
# Partially parsed test_map_structure_zip_with_custom_function. Retrieved 7/10 statements.
# Partially parsed test_map_structure_zip_preserves_tuple_type. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_4, var_8]

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'unordered'

def test_case_0():
    var_0 = 'items'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = 5
    var_10 = 6
    var_11 = (var_9, var_10)
    var_12 = 7
    var_13 = 8
    var_14 = (var_12, var_13)
    var_15 = [var_11, var_14]
    var_16 = {var_0: var_15}
    var_17 = [var_8, var_16]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'c'
    var_4 = 'd'
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = [var_3]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_complex_nested_structure. Retrieved 16/19 statements.
# Partially parsed test_map_structure_zip_with_empty_list. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_empty_dict. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 3
    var_10 = {var_2: var_9}
    var_11 = 4
    var_12 = {var_5: var_11}
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = [var_8, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 4
    var_8 = 5
    var_9 = [var_7, var_8]
    var_10 = 6
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = [var_6, var_11]

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = {var_4, var_5, var_6}
    var_8 = [var_3, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'Structures cannot contain `set`'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 'data'
    var_1 = 'value'
    var_2 = 1
    var_3 = 'nested'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = 4
    var_10 = 5
    var_11 = {var_3: var_10}
    var_12 = [var_9, var_11]
    var_13 = 6
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = [var_8, var_14]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = [var_0, var_1]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 4/26 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = '__type_check__'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = False
    assert var_0 is False



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_map_structure_tuple_predicate. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_false. Retrieved 16/37 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = 3
    var_9 = 4
    var_10 = [var_8, var_9]
    var_11 = [var_7, var_10]
    var_12 = lambda *args: sum(args)
    var_13 = module_0.map_structure_zip(var_12, var_11)
    var_14 = 0
    var_15 = var_11[var_14]
    var_16 = var_15.__class__
    var_17 = hasattr(var_15, var_4)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 5/31 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = None
    var_6 = 'no_type_check'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 3/25 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__no_type_check__'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 7/13 statements.
# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 8/17 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_structure. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_mixed_nested. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_scalar. Retrieved 4/10 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_complex_function. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_preserves_tuple_structure. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = (var_1, var_2, var_4)
    var_6 = [var_3, var_5]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_4, var_8]

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = {var_4, var_5, var_6}
    var_8 = [var_3, var_7]
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'set'
    var_11 = bool('set' in str(e).lower())
    assert var_11 is True

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 4
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_preserves_list_type. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_empty_collections. Retrieved 5/11 statements.
# Partially parsed test_map_structure_with_string_keys_in_dict. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = ()
    var_2 = {}
    var_3 = set()
    var_4 = set()

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_false. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_type_check__'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_map_structure_tuple_predicate. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/18 statements.
# Partially parsed test_map_structure_zip_preserves_dict_type. Retrieved 17/25 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [5, 7, 9])
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = (var_5, var_6, var_7)
    var_9 = (var_4, var_8)
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == (5, 7, 9))
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_3, var_5]
    var_7 = 4
    var_8 = 5
    var_9 = [var_7, var_8]
    var_10 = 6
    var_11 = [var_10]
    var_12 = [var_9, var_11]
    var_13 = [var_6, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == [[5, 7], [9]])
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': {'x': 3}})
    assert var_11 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 4
    var_9 = 5
    var_10 = [var_8, var_9]
    var_11 = 6
    var_12 = {var_1: var_10, var_2: var_11}
    var_13 = [var_7, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == {'a': [5, 7], 'b': 9})
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 50

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'helloworld'

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'Structures cannot contain `set`'

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 3
    var_9 = (var_1, var_8)
    var_10 = 4
    var_11 = (var_4, var_10)
    var_12 = [var_9, var_11]
    var_13 = (var_1, var_10)
    var_14 = 6
    var_15 = (var_4, var_14)
    var_16 = [var_13, var_15]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = {}
    var_2 = {}
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 8/16 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dict. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_mixed_list_dict. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_complex_nested_structure. Retrieved 17/20 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = [var_1, var_3]
    var_8 = 5
    var_9 = [var_4, var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = 3
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 4
    var_6 = 5
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 3
    var_10 = {var_2: var_9}
    var_11 = 4
    var_12 = {var_5: var_11}
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = [var_8, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = {var_0: var_6}
    var_8 = 4
    var_9 = {var_0: var_8}
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'set'
    var_7 = bool('set' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 'list'
    var_1 = 'value'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = 10
    var_10 = 20
    var_11 = 30
    var_12 = (var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = 40
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = [var_8, var_15]



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = False



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_map_structure_decorator_exists. Retrieved 3/23 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__wrapped__'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_false. Retrieved 5/28 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = ()
    var_5 = '__no_map__'
    var_6 = '__no_type_check__'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_with_nested_namedtuple. Retrieved 12/24 statements.
# Partially parsed test_map_structure_zip_with_multiple_args. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 3
    var_10 = {var_2: var_9}
    var_11 = 4
    var_12 = {var_5: var_11}
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = [var_8, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 4
    var_8 = 5
    var_9 = [var_7, var_8]
    var_10 = 6
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = [var_6, var_11]

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 5
    var_9 = 6
    var_10 = 7
    var_11 = 9

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Structures cannot contain `set`'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 5/29 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = '__wrapped__'
    var_6 = 'map_structure_zip'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/18 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_mixed_structure. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_custom_function. Retrieved 9/15 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_4, var_8]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'set'
    var_7 = bool('set' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_map_structure_zip_dict. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_map_structure_zip_with_list. Retrieved 13/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_3, var_7, var_11]



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_named_tuples. Retrieved 9/18 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_scalar_values. Retrieved 3/9 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.
# Partially parsed test_map_structure_zip_with_complex_nested_structure. Retrieved 14/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_4, var_8]

def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Structures cannot contain `set`'

def test_case_0():
    var_0 = 'data'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_2, var_3)
    var_5 = [var_1, var_4]
    var_6 = {var_0: var_5}
    var_7 = 4
    var_8 = 5
    var_9 = 6
    var_10 = (var_8, var_9)
    var_11 = [var_7, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_6, var_12]



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_map_structure_zip_with_list. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_evaluates_to_false. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_7]
    var_9 = 0
    var_10 = var_8[var_9]
    var_11 = '__no_map_structure__'
    var_12 = var_10.__class__
    var_13 = hasattr(var_10, var_11)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_list_of_dicts. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_dict_of_lists. Retrieved 12/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = [var_1, var_3]
    var_8 = 5
    var_9 = [var_4, var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_4, var_8]

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = {var_0: var_6}
    var_8 = 4
    var_9 = {var_0: var_8}
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 4
    var_7 = 5
    var_8 = 6
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_0: var_9}
    var_11 = [var_5, var_10]



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_map_structure_zip_simple_list. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_nested_list. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_tuple. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_nested_dict. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_mixed_list_dict. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_scalar_values. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_three_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_empty_list. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_set_raises_error. Retrieved 7/11 statements.
# Partially parsed test_map_structure_zip_string_no_mapping. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_complex_nested_structure. Retrieved 16/19 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 3
    var_8 = {var_2: var_7}
    var_9 = 4
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = {var_0: var_6}
    var_8 = 4
    var_9 = {var_0: var_8}
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'set'
    var_9 = bool('set' in str(e).lower())
    assert var_9 is True

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 'data'
    var_1 = 'nested'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'value'
    var_6 = 10
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 3
    var_10 = 4
    var_11 = [var_9, var_10]
    var_12 = 20
    var_13 = {var_5: var_12}
    var_14 = {var_0: var_11, var_1: var_13}
    var_15 = [var_8, var_14]



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/20 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dict. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_mixed_structure. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 4/10 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.
# Partially parsed test_map_structure_zip_with_custom_callable. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_single_collection. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6
    var_9 = '_fields'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 3
    var_10 = {var_2: var_9}
    var_11 = 4
    var_12 = {var_5: var_11}
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = [var_8, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_4, var_8]

def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Structures cannot contain `set`'

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 2
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 7/13 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_list_of_dicts. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_with_scalar_values. Retrieved 4/10 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_complex_nested_structure. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.
# Partially parsed test_map_structure_zip_with_multiple_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_empty_list. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_preserves_tuple_structure. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = (var_1, var_2, var_4)
    var_6 = [var_3, var_5]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = 20
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = {var_0: var_6}
    var_8 = 4
    var_9 = {var_0: var_8}
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'items'
    var_1 = 'value'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 10
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 3
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = 20
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = [var_6, var_11]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'set'
    var_7 = bool('set' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 10
    var_4 = 20
    var_5 = [var_3, var_4]
    var_6 = 100
    var_7 = 200
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = [var_3]



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_preserves_tuple_type. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x ** var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = bool(var_5 == [1, 4, 9])
    assert var_6 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 17/20 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/18 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = [var_6, var_7]
    var_9 = 6
    var_10 = [var_9]
    var_11 = [var_8, var_10]
    var_12 = [var_5, var_11]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = (var_3,)
    var_5 = (var_2, var_4)
    var_6 = 4
    var_7 = 5
    var_8 = (var_6, var_7)
    var_9 = 6
    var_10 = (var_9,)
    var_11 = (var_8, var_10)
    var_12 = [var_5, var_11]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 3
    var_8 = {var_2: var_7}
    var_9 = 4
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 5
    var_10 = 6
    var_11 = [var_9, var_10]
    var_12 = 7
    var_13 = 8
    var_14 = (var_12, var_13)
    var_15 = {var_0: var_11, var_1: var_14}
    var_16 = [var_8, var_15]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'set'
    var_7 = bool('set' in str(e).lower())
    assert var_7 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 4/10 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/18 statements.
# Partially parsed test_map_structure_zip_with_complex_nested_structure. Retrieved 16/19 statements.
# Partially parsed test_map_structure_zip_with_empty_list. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_empty_dict. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.
# Partially parsed test_map_structure_zip_with_custom_function. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = {var_0: var_6}
    var_8 = 4
    var_9 = {var_0: var_8}
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]

def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'data'
    var_1 = 'count'
    var_2 = 1
    var_3 = 'val'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = 4
    var_10 = 5
    var_11 = {var_3: var_10}
    var_12 = [var_9, var_11]
    var_13 = 6
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = [var_8, var_14]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'unordered'

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 2
    var_5 = 4
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_structure. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_three_arguments. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.
# Partially parsed test_map_structure_zip_with_complex_nested_structure. Retrieved 17/20 statements.
# Partially parsed test_map_structure_zip_empty_lists. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_empty_dict. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_4, var_8]

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'set'
    var_9 = bool('set' in str(e).lower())
    assert var_9 is True

def test_case_0():
    var_0 = 'data'
    var_1 = 'meta'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 10
    var_6 = 20
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 3
    var_10 = 4
    var_11 = [var_9, var_10]
    var_12 = 30
    var_13 = 40
    var_14 = (var_12, var_13)
    var_15 = {var_0: var_11, var_1: var_14}
    var_16 = [var_8, var_15]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = [var_0, var_1]



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_map_structure_zip_with_tuple. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 5/26 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = ()
    var_5 = '__no_map__'
    var_6 = None



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__wrapped__'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_map_structure_zip_list_predicate. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_no_type_check_predicate_is_false. Retrieved 5/26 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = ()
    var_5 = 'no_map'
    var_6 = '@no_type_check'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_map_structure_zip_set_raises_value_error. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_set_raises_value_error. Retrieved 9/32 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = ()
    var_5 = '__no_map__'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_6, var_7, var_8}
    var_10 = [var_9]
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 3/21 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_type_check__'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 25/28 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'dict'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_3, var_4, var_7]
    var_9 = 5
    var_10 = 6
    var_11 = 7
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = 'nested'
    var_15 = 8
    var_16 = {var_14: var_15}
    var_17 = {var_0: var_8, var_1: var_13, var_2: var_16}
    var_18 = [var_6, var_9]
    var_19 = [var_4, var_5, var_18]
    var_20 = (var_11, var_15)
    var_21 = (var_10, var_20)
    var_22 = 9
    var_23 = {var_14: var_22}
    var_24 = {var_0: var_19, var_1: var_21, var_2: var_23}



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_map_structure_zip_with_tuple. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 1/8 statements.


def test_case_0():
    var_0 = '__no_type_check__'
    var_1 = '@no_type_check'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/18 statements.
# Partially parsed test_map_structure_zip_with_nested_namedtuple. Retrieved 9/21 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_preserves_dict_structure. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = {var_0: var_6}
    var_8 = 4
    var_9 = {var_0: var_8}
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]

def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Structures cannot contain `set`'

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda a, b: a * b
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = [var_2, var_3, var_5]
    var_7 = [var_4, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == [2, 6, 12])
    assert var_9 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 4
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_7}
    var_9 = [var_6, var_8]



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_false. Retrieved 5/28 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map_structure__'
    var_5 = '__no_type_check__'
    var_6 = 'map_structure_zip'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_map_structure_with_simple_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 11/14 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_deeply_nested. Retrieved 11/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'd'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 3
    var_8 = {var_0: var_6, var_1: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = 'b'
    var_6 = 3
    var_7 = 4
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = [var_4, var_9]

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

import flutes.structure as module_0

def test_case_0():
    var_0 = 100
    var_1 = lambda x: x + var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.map_structure(var_1, var_5)
    var_7 = bool(var_6 == [101, 102, 103])
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = [var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = [var_8]
    var_10 = [var_4, var_9]



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_no_type_check_predicate_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = True
    var_1 = True
    var_2 = False



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '__no_type_check__'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_map_structure_zip_with_no_type_check_decorator. Retrieved 27/51 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__dataclass_fields__'
    var_5 = '__wrapped__'
    var_6 = lambda x, y: x + y
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = 3
    var_11 = 4
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = module_0.map_structure_zip(var_6, var_13)
    var_15 = bool(var_14 == [4, 6])
    assert var_15 is True
    var_16 = (var_7, var_8)
    var_17 = (var_10, var_11)
    var_18 = [var_16, var_17]
    var_19 = module_0.map_structure_zip(var_6, var_18)
    var_20 = bool(var_19 == (4, 6))
    assert var_20 is True
    var_21 = 'a'
    var_22 = {var_21: var_7}
    var_23 = {var_21: var_8}
    var_24 = [var_22, var_23]
    var_25 = module_0.map_structure_zip(var_6, var_24)
    var_26 = bool(var_25 == {'a': 3})
    assert var_26 is True
    var_27 = lambda x, y: x + y
    var_28 = 'hello'
    var_29 = 'world'
    var_30 = [var_28, var_29]
    var_31 = module_0.map_structure_zip(var_27, var_30)
    assert var_31 == 'helloworld'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_map_structure_with_simple_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_tuple. Retrieved 1/4 statements.
# Partially parsed test_map_structure_preserves_dict_type. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_no_type_check_predicate_evaluates_to_false. Retrieved 1/19 statements.


def test_case_0():
    var_0 = '__no_type_check__'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_evaluates_to_false. Retrieved 28/36 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = [var_6, var_9]
    var_11 = [var_4, var_5]
    var_12 = [var_11]
    var_13 = [var_7, var_8]
    var_14 = [var_13]
    var_15 = [var_12, var_14]
    var_16 = lambda x, y: x + y
    var_17 = 'a'
    var_18 = 'b'
    var_19 = {var_17: var_4, var_18: var_5}
    var_20 = 10
    var_21 = 20
    var_22 = {var_17: var_20, var_18: var_21}
    var_23 = [var_19, var_22]
    var_24 = module_0.map_structure_zip(var_16, var_23)
    var_25 = bool(var_24 == {'a': 11, 'b': 22})
    assert var_25 is True
    var_26 = lambda x, y: x + y
    var_27 = (var_4, var_5)
    var_28 = (var_7, var_8)
    var_29 = [var_27, var_28]
    var_30 = module_0.map_structure_zip(var_26, var_29)
    var_31 = bool(var_30 == (4, 6))
    assert var_31 is True



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 7/30 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = '__wrapped__'
    var_6 = '__no_type_check__'
    var_7 = False
    var_8 = True



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_string_transformation. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_tuple. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 3
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()



# Parsed testcases at query #91
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/9 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 13/16 statements.
# Partially parsed test_map_structure_with_empty_collections. Retrieved 5/11 statements.
# Partially parsed test_map_structure_preserves_structure_type. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'nested'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 'value'
    var_10 = 5
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}

def test_case_0():
    var_0 = []
    var_1 = ()
    var_2 = {}
    var_3 = set()
    var_4 = set()

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = (var_0, var_1, var_2)
    var_5 = 'a'
    var_6 = {var_5: var_0}



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_map_structure_decorator_exists. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_false. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1. Retrieved 4/28 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = '__wrapped__'



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 5/31 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = '__no_type_check__'
    var_6 = True



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 6/9 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/14 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_decorator_exists. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

import flutes.structure as module_0

def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = 2
    var_5 = lambda x: x * var_4
    var_6 = 1
    var_7 = 3
    var_8 = [var_6, var_4, var_7]
    var_9 = module_0.map_structure(var_5, var_8)
    var_10 = bool(var_9 == [2, 4, 6])
    assert var_10 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_structure. Retrieved 13/16 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_complex_nested_structure. Retrieved 9/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'dict'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 'nested'
    var_10 = 5
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 3
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'a'
    var_1 = -1
    var_2 = -2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = -3
    var_6 = -4
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_named_tuples. Retrieved 8/17 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_mixed_nested_structure. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 12/20 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.
# Partially parsed test_map_structure_zip_with_multiple_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_empty_list. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_complex_nested_structure. Retrieved 17/20 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = [var_1, var_3]
    var_8 = 5
    var_9 = [var_4, var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'hello'
    var_1 = ' world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = [var_0, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = {var_1: var_6}
    var_8 = [var_5, var_7]
    var_9 = [var_4, var_8]

def test_case_0():
    var_0 = 'a'
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = (var_0, var_7)
    var_9 = 5
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'set'
    var_9 = bool('set' in str(e).lower())
    assert var_9 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'data'
    var_1 = 'value'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 4
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = 5
    var_10 = 6
    var_11 = 7
    var_12 = (var_10, var_11)
    var_13 = [var_9, var_12]
    var_14 = 8
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = [var_8, var_15]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 10
    var_6 = 20
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_string_conversion. Retrieved 4/7 statements.
# Partially parsed test_map_structure_preserves_tuple_type. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 13/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'dict'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 'nested'
    var_10 = 5
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_15_true. Retrieved 14/37 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '_no_map'
    var_5 = 2
    var_6 = lambda x: x * var_5
    var_7 = 42
    var_8 = [var_7]
    var_9 = module_0.map_structure_zip(var_6, var_8)
    assert var_9 == 84
    var_10 = True
    var_11 = lambda x: x
    var_12 = lambda x: x.upper()
    var_13 = 'hello'
    var_14 = [var_13]
    var_15 = module_0.map_structure_zip(var_12, var_14)
    assert var_15 == 'HELLO'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 4/24 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = ()
    var_5 = '__no_map__'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_map_structure_dict_predicate. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '\n'
    var_5 = 0



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_map_structure_zip_with_tuple. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = (var_6,)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_evaluates_to_false. Retrieved 9/30 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '_no_map_'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = var_8.__class__
    var_10 = hasattr(var_8, var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_map_structure_list_predicate. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 11/14 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 8/17 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_three_arguments. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = [var_1, var_1]
    var_8 = [var_1, var_1]
    var_9 = [var_7, var_8]
    var_10 = [var_6, var_9]

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = 30
    var_3 = (var_0, var_1, var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = 3
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 4
    var_6 = 5
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 4
    var_8 = 5
    var_9 = [var_7, var_8]
    var_10 = 6
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = [var_6, var_11]

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'set'
    var_7 = bool('set' in str(e).lower())
    assert var_7 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_map_structure_with_simple_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_deeply_nested_structure. Retrieved 11/14 statements.
# Partially parsed test_map_structure_with_single_element. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_preserves_nested_tuple_structure. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 'b'
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = {var_5: var_8}
    var_10 = [var_4, var_9]

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 3
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_map_structure_zip_with_list. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dict. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_dict_and_list. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_custom_function. Retrieved 7/13 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_complex_nested_structure. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_single_element_lists. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 3
    var_8 = {var_2: var_7}
    var_9 = 4
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_4, var_8]

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 15
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = [var_0, var_1]
    var_3 = 4
    var_4 = 5
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Structures cannot contain `set`'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 'data'
    var_1 = 1
    var_2 = 'val'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = 4
    var_9 = {var_2: var_8}
    var_10 = [var_7, var_9]
    var_11 = {var_0: var_10}
    var_12 = [var_6, var_11]

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(var_8 == [8, 15])
    assert var_9 is True

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = [var_2]
    var_4 = 3
    var_5 = [var_4]
    var_6 = [var_1, var_3, var_5]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1. Retrieved 3/21 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_type_check__'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_map_structure_zip_set_raises_value_error. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 13/16 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_tuple. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_function_converting_type. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'data'
    var_1 = 'nested'
    var_2 = 'single'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 'values'
    var_7 = 3
    var_8 = 4
    var_9 = (var_7, var_8)
    var_10 = {var_6: var_9}
    var_11 = 5
    var_12 = {var_0: var_5, var_1: var_10, var_2: var_11}

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 6/13 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 13/16 statements.
# Partially parsed test_map_structure_preserves_dict_type. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_tuple. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'dict'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 'nested'
    var_10 = 5
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_map_structure_zip_with_list. Retrieved 13/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_3, var_7, var_11]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '_no_type_check'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_map_structure_zip_with_tuple. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_map_structure_zip_predicate. Retrieved 13/31 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = ()
    var_5 = '__no_map__'
    var_6 = lambda x, y: x + y
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = 3
    var_11 = 4
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = module_0.map_structure_zip(var_6, var_13)
    var_15 = bool(var_14 == [4, 6])
    assert var_15 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 6/30 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = '__wrapped__'
    var_6 = None
    var_7 = False



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 6/9 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/14 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 6/13 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_preserves_empty_collections. Retrieved 3/8 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 11/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = ()

def test_case_0():
    var_0 = 'nums'
    var_1 = 'nested'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'values'
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = {var_5: var_8}
    var_10 = {var_0: var_4, var_1: var_9}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_map_structure_zip_with_tuple. Retrieved 15/20 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = (var_9, var_12)
    var_14 = (var_6, var_13)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/18 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 17/25 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [5, 7, 9])
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = 5
    var_9 = 6
    var_10 = [var_8, var_9]
    var_11 = 7
    var_12 = 8
    var_13 = [var_11, var_12]
    var_14 = [var_10, var_13]
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[6, 8], [10, 12]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = (var_5, var_6, var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == (5, 7, 9))
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = 5
    var_9 = 6
    var_10 = (var_8, var_9)
    var_11 = 7
    var_12 = 8
    var_13 = (var_11, var_12)
    var_14 = (var_10, var_13)
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == ((6, 8), (10, 12)))
    assert var_17 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'x'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 'y'
    var_7 = 2
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 3
    var_11 = {var_3: var_10}
    var_12 = 4
    var_13 = {var_6: var_12}
    var_14 = {var_1: var_11, var_2: var_13}
    var_15 = [var_9, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == {'a': {'x': 4}, 'b': {'y': 6}})
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 5
    var_11 = 6
    var_12 = [var_10, var_11]
    var_13 = 7
    var_14 = 8
    var_15 = (var_13, var_14)
    var_16 = {var_1: var_12, var_2: var_15}
    var_17 = [var_9, var_16]
    var_18 = module_0.map_structure_zip(var_0, var_17)
    var_19 = bool(var_18 == {'a': [6, 8], 'b': (10, 12)})
    assert var_19 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 15

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'helloworld'

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = 3
    var_5 = 4
    var_6 = {var_4, var_5}
    var_7 = [var_3, var_6]
    var_8 = module_0.map_structure_zip(var_0, var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'set'
    var_11 = bool('set' in str(e).lower())
    assert var_11 is True

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = 3
    var_9 = (var_1, var_8)
    var_10 = 4
    var_11 = (var_4, var_10)
    var_12 = [var_9, var_11]
    var_13 = (var_1, var_10)
    var_14 = 6
    var_15 = (var_4, var_14)
    var_16 = [var_13, var_15]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/18 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_three_objects. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.
# Partially parsed test_map_structure_zip_preserves_list_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_empty_list. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_empty_dict. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = [var_6, var_7]
    var_9 = 6
    var_10 = [var_9]
    var_11 = [var_8, var_10]
    var_12 = [var_5, var_11]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = (var_3, var_7)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = (var_3,)
    var_5 = (var_2, var_4)
    var_6 = 4
    var_7 = 5
    var_8 = (var_6, var_7)
    var_9 = 6
    var_10 = (var_9,)
    var_11 = (var_8, var_10)
    var_12 = (var_5, var_11)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 3
    var_6 = [var_5]
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = 4
    var_9 = 5
    var_10 = (var_8, var_9)
    var_11 = 6
    var_12 = [var_11]
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = [var_7, var_13]

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'set'

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = [var_0, var_1, var_2]
    var_4 = 5
    var_5 = 6
    var_6 = 7
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = [var_0, var_1]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_map_structure_with_simple_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_preserves_ordered_dict. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_deeply_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_empty_collections. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'outer'
    var_1 = 'inner'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}

def test_case_0():
    var_0 = []
    var_1 = ()
    var_2 = {}
    var_3 = set()
    var_4 = set()



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 10/18 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == [5, 7, 9])
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = (var_5, var_6, var_7)
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == (5, 7, 9))
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': 4, 'b': 6})
    assert var_11 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = [var_3, var_6]
    var_8 = 5
    var_9 = 6
    var_10 = [var_8, var_9]
    var_11 = 7
    var_12 = 8
    var_13 = [var_11, var_12]
    var_14 = [var_10, var_13]
    var_15 = [var_7, var_14]
    var_16 = module_0.map_structure_zip(var_0, var_15)
    var_17 = bool(var_16 == [[6, 8], [10, 12]])
    assert var_17 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2
    var_7 = {var_2: var_6}
    var_8 = {var_1: var_7}
    var_9 = [var_5, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = bool(var_10 == {'a': {'b': 3}})
    assert var_11 is True

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = 4
    var_9 = 6

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x * y
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 50

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    assert var_4 == 'helloworld'

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = {var_1: var_5, var_2: var_6}
    var_8 = 4
    var_9 = 5
    var_10 = [var_8, var_9]
    var_11 = 6
    var_12 = {var_1: var_10, var_2: var_11}
    var_13 = [var_7, var_12]
    var_14 = module_0.map_structure_zip(var_0, var_13)
    var_15 = bool(var_14 == {'a': [5, 7], 'b': 9})
    assert var_15 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y, z: x + y + z
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.map_structure_zip(var_0, var_10)
    var_12 = bool(var_11 == [9, 12])
    assert var_12 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = {var_1, var_2, var_3}
    var_5 = [var_4]
    var_6 = module_0.map_structure_zip(var_0, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'unordered'

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = []
    var_2 = []
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True

import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = {}
    var_2 = {}
    var_3 = [var_1, var_2]
    var_4 = module_0.map_structure_zip(var_0, var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_line_1_evaluates_to_false. Retrieved 20/33 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = lambda x: x
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = [var_9]
    var_11 = module_0.map_structure_zip(var_5, var_10)
    assert var_11 is False
    var_12 = lambda x: x
    var_13 = 'a'
    var_14 = {var_13: var_6}
    var_15 = [var_14]
    var_16 = module_0.map_structure_zip(var_12, var_15)
    assert var_16 is False
    var_17 = lambda x: x
    var_18 = (var_6, var_7)
    var_19 = [var_18]
    var_20 = module_0.map_structure_zip(var_17, var_19)
    assert var_20 is False
    var_21 = lambda x: x



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_evaluates_to_false. Retrieved 12/30 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = ()
    var_5 = '_no_map_'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = var_9.__class__
    var_11 = var_10 in var_4
    var_12 = hasattr(var_9, var_5)
    var_13 = var_11 or var_12
    assert var_13 is False



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_map_structure_zip_set_raises_value_error. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_map_structure_predicate_line_1_false. Retrieved 10/32 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = lambda x: x * var_6
    var_10 = var_8.__class__
    var_11 = hasattr(var_8, var_4)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 6/27 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = ()
    var_5 = '_no_map'
    var_6 = '__no_type_check__'
    var_7 = False



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_preserves_structure_with_list_of_dicts. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_identity_function. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = 3
    var_5 = 4
    var_6 = (var_4, var_5)
    var_7 = [var_0, var_3, var_6]



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_map_structure_zip_with_list. Retrieved 13/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_3, var_7, var_11]



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 6/13 statements.
# Partially parsed test_map_structure_with_nested_namedtuple. Retrieved 7/17 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_string_transformation. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_set_raises_value_error. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_false. Retrieved 18/23 statements.


def test_case_0():
    var_0 = 'typing'
    var_1 = __import__(var_0)
    var_2 = 'no_type_check'
    var_3 = hasattr(var_1, var_2)
    var_4 = __import__(var_0)
    var_5 = var_4.no_type_check
    var_6 = None
    var_7 = var_5 if var_3 else var_6
    var_8 = 'R'
    var_9 = []
    var_10 = 'T'
    var_11 = []
    var_12 = 1
    var_13 = 2
    var_14 = [var_12, var_13]
    var_15 = 3
    var_16 = 4
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = True
    assert var_19 is True
    var_20 = bool(not False)
    assert var_20 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_map_structure_zip_dict_predicate. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 5/28 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = ()
    var_5 = '__no_map__'
    var_6 = '__wrapped__'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_nested_namedtuple. Retrieved 10/21 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 13/16 statements.
# Partially parsed test_map_structure_with_function_transformation. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6
    var_9 = 8

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'dict'
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = 3
    var_7 = 4
    var_8 = (var_6, var_7)
    var_9 = 'nested'
    var_10 = 5
    var_11 = {var_9: var_10}
    var_12 = {var_0: var_5, var_1: var_8, var_2: var_11}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_map_structure_zip_predicate. Retrieved 22/43 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = ()
    var_5 = '__no_map__'
    var_6 = lambda x, y: x + y
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = 3
    var_11 = 4
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = module_0.map_structure_zip(var_6, var_13)
    var_15 = bool(var_14 == [4, 6])
    assert var_15 is True
    var_16 = (var_7, var_8)
    var_17 = (var_10, var_11)
    var_18 = [var_16, var_17]
    var_19 = module_0.map_structure_zip(var_6, var_18)
    var_20 = bool(var_19 == (4, 6))
    assert var_20 is True
    var_21 = 'a'
    var_22 = {var_21: var_7}
    var_23 = {var_21: var_8}
    var_24 = [var_22, var_23]
    var_25 = module_0.map_structure_zip(var_6, var_24)
    var_26 = bool(var_25 == {'a': 3})
    assert var_26 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 3/24 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__no_type_check__'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_map_structure_tuple_predicate. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 11/17 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = lambda x, y: x + y
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = 3
    var_9 = 4
    var_10 = [var_8, var_9]
    var_11 = [var_7, var_10]
    var_12 = module_0.map_structure_zip(var_4, var_11)
    var_13 = bool(var_12 == [4, 6])
    assert var_13 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_false. Retrieved 5/27 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = ()
    var_5 = '__no_map__'
    var_6 = '__no_type_check__'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/18 statements.
# Partially parsed test_map_structure_zip_with_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dict. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_mixed_structure. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 4/10 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.
# Partially parsed test_map_structure_zip_with_complex_nested_structure. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_custom_function. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 3
    var_10 = {var_2: var_9}
    var_11 = 4
    var_12 = {var_5: var_11}
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = [var_8, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = 4
    var_8 = 5
    var_9 = [var_7, var_8]
    var_10 = 6
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = [var_6, var_11]

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Structures cannot contain `set`'

def test_case_0():
    var_0 = 'data'
    var_1 = 'val'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = 3
    var_9 = {var_1: var_8}
    var_10 = 4
    var_11 = {var_1: var_10}
    var_12 = [var_9, var_11]
    var_13 = {var_0: var_12}
    var_14 = [var_7, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'x'
    var_4 = 'y'
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_map_structure_no_type_check_decorator_present. Retrieved 3/25 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__no_type_check__'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_nested_structures. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'hello'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 6/30 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = '__wrapped__'
    var_6 = None
    var_7 = False



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_evaluates_to_false. Retrieved 13/33 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = 3
    var_9 = 4
    var_10 = [var_8, var_9]
    var_11 = [var_7, var_10]
    var_12 = lambda *args: sum(args)
    var_13 = [var_11]
    var_14 = module_0.map_structure_zip(var_12, var_13)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1. Retrieved 12/32 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '_no_map_'
    var_5 = lambda x, y: x + y
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = 3
    var_10 = 4
    var_11 = [var_9, var_10]
    var_12 = [var_8, var_11]
    var_13 = module_0.map_structure_zip(var_5, var_12)
    var_14 = bool(var_13 == [4, 6])
    assert var_14 is True



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_function_returning_string. Retrieved 4/7 statements.
# Partially parsed test_map_structure_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_empty_dict. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = (var_0, var_3)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_map_structure_predicate_line_1_false. Retrieved 8/28 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = lambda x: x * var_5
    var_9 = module_0.map_structure(var_8, var_7)
    var_10 = bool(var_9 == [2, 4, 6])
    assert var_10 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 19/22 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_namedtuples. Retrieved 12/22 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_mixed_structures. Retrieved 14/17 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 22/32 statements.
# Partially parsed test_map_structure_zip_with_empty_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.
# Partially parsed test_map_structure_zip_with_custom_function. Retrieved 9/15 statements.
# Partially parsed test_map_structure_zip_with_single_collection. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = [var_8, var_9, var_10]
    var_12 = [var_3, var_7, var_11]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = [var_6, var_7]
    var_9 = 6
    var_10 = [var_9]
    var_11 = [var_8, var_10]
    var_12 = 7
    var_13 = 8
    var_14 = [var_12, var_13]
    var_15 = 9
    var_16 = [var_15]
    var_17 = [var_14, var_16]
    var_18 = [var_5, var_11, var_17]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = 7
    var_9 = 8
    var_10 = 9
    var_11 = (var_8, var_9, var_10)
    var_12 = [var_3, var_7, var_11]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 5
    var_9 = 6
    var_10 = 9
    var_11 = 12

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = 5
    var_9 = 6
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = [var_4, var_7, var_10]

def test_case_0():
    var_0 = 'a'
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 3
    var_9 = {var_1: var_8}
    var_10 = {var_0: var_9}
    var_11 = [var_4, var_7, var_10]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = 5
    var_10 = 6
    var_11 = [var_9, var_10]
    var_12 = {var_0: var_11}
    var_13 = [var_4, var_8, var_12]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = 5
    var_13 = (var_0, var_12)
    var_14 = 6
    var_15 = (var_3, var_14)
    var_16 = [var_13, var_15]
    var_17 = 9
    var_18 = (var_0, var_17)
    var_19 = 12
    var_20 = (var_3, var_19)
    var_21 = [var_18, var_20]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Structures cannot contain `set`'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_1, var_2, var_4]
    var_6 = 5
    var_7 = [var_2, var_4, var_6]
    var_8 = [var_3, var_5, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/9 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_string_function. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 3
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 6/9 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/14 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/9 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_string_function. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_tuple. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()



# Parsed testcases at query #73
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 4/26 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = '__wrapped__'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_false. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_type_check__'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = [var_6, var_9]



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_decorator_exists. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 24/27 statements.
# Partially parsed test_map_structure_preserves_tuple_type. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_tuple. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'y'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'dict'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_3, var_4, var_7]
    var_9 = 5
    var_10 = 6
    var_11 = (var_9, var_10)
    var_12 = 'x'
    var_13 = 'y'
    var_14 = 7
    var_15 = 8
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = {var_0: var_8, var_1: var_11, var_2: var_16}
    var_18 = [var_6, var_9]
    var_19 = [var_4, var_5, var_18]
    var_20 = (var_10, var_14)
    var_21 = 9
    var_22 = {var_12: var_15, var_13: var_21}
    var_23 = {var_0: var_19, var_1: var_20, var_2: var_22}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_false. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = ()
    var_5 = '_no_map'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_structure. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_with_scalar_values. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_custom_function. Retrieved 7/13 statements.
# Partially parsed test_map_structure_zip_with_three_collections. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_empty_list. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_empty_dict. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.
# Partially parsed test_map_structure_zip_with_complex_nested_structure. Retrieved 13/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_4, var_8]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 15
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = [var_1, var_2, var_4]
    var_6 = [var_3, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Structures cannot contain `set`'

def test_case_0():
    var_0 = 'data'
    var_1 = 1
    var_2 = 'val'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = [var_1, var_4]
    var_6 = {var_0: var_5}
    var_7 = 3
    var_8 = 4
    var_9 = {var_2: var_8}
    var_10 = [var_7, var_9]
    var_11 = {var_0: var_10}
    var_12 = [var_6, var_11]



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 3/10 statements.


def test_case_0():
    var_0 = '\n'
    var_1 = 0
    var_2 = '@no_type_check'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 6/13 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/9 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_empty_collections. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = ()
    var_3 = set()
    var_4 = set()



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_no_type_check_decorator_evaluates_to_false. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = None
    var_5 = var_4 is not None
    assert var_5 is False



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 3/25 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_type_check__'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 27/38 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = 1
    var_5 = 2
    var_6 = [var_4, var_5]
    var_7 = 3
    var_8 = 4
    var_9 = [var_7, var_8]
    var_10 = [var_6, var_9]
    var_11 = (var_4, var_5)
    var_12 = (var_7, var_8)
    var_13 = [var_11, var_12]
    var_14 = 'a'
    var_15 = 'b'
    var_16 = {var_14: var_4, var_15: var_5}
    var_17 = {var_14: var_7, var_15: var_8}
    var_18 = [var_16, var_17]
    var_19 = [var_4, var_5]
    var_20 = [var_19]
    var_21 = [var_7, var_8]
    var_22 = [var_21]
    var_23 = [var_20, var_22]
    var_24 = lambda x, y: x + y
    var_25 = 5
    var_26 = 10
    var_27 = [var_25, var_26]
    var_28 = module_0.map_structure_zip(var_24, var_27)
    assert var_28 == 15
    var_29 = '@no_type_check'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/9 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 11/14 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_tuple. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_complex_function. Retrieved 8/11 statements.
# Partially parsed test_map_structure_preserves_dict_type. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 'c'
    var_7 = 4
    var_8 = {var_6: var_7}
    var_9 = (var_5, var_8)
    var_10 = {var_0: var_4, var_1: var_9}

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_map_structure_zip_with_decorator. Retrieved 4/12 statements.


def test_case_0():
    var_0 = '__wrapped__'
    var_1 = '__decorators__'
    var_2 = []
    var_3 = 'no_type_check'
    var_4 = '@no_type_check'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_identity_function. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_string_function. Retrieved 4/7 statements.
# Partially parsed test_map_structure_preserves_empty_collections. Retrieved 5/11 statements.
# Partially parsed test_map_structure_with_bool_function. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 'a'
    var_5 = 4
    var_6 = {var_4: var_5}
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = ()
    var_3 = set()
    var_4 = set()

def test_case_0():
    var_0 = 1
    var_1 = -2
    var_2 = 3
    var_3 = -4
    var_4 = [var_0, var_1, var_2, var_3]



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/13 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 7/13 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_nested_structure. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/19 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/9 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.
# Partially parsed test_map_structure_zip_with_empty_list. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_empty_dict. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_complex_nested_structure. Retrieved 17/20 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'x'
    var_3 = 'y'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'p'
    var_6 = 'q'
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = [var_3]
    var_5 = [var_2, var_4]
    var_6 = 4
    var_7 = 5
    var_8 = [var_6, var_7]
    var_9 = 6
    var_10 = [var_9]
    var_11 = [var_8, var_10]
    var_12 = [var_5, var_11]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 2
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 3
    var_7 = {var_0: var_6}
    var_8 = 4
    var_9 = {var_0: var_8}
    var_10 = [var_7, var_9]
    var_11 = [var_5, var_10]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'unordered'

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 5
    var_10 = 6
    var_11 = [var_9, var_10]
    var_12 = 7
    var_13 = 8
    var_14 = (var_12, var_13)
    var_15 = {var_0: var_11, var_1: var_14}
    var_16 = [var_8, var_15]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_mixed_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_ordered_dict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 10/13 statements.
# Partially parsed test_map_structure_with_string_function. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 'd'
    var_5 = 2
    var_6 = 3
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_0: var_2, var_1: var_7}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {var_1: var_6}
    var_8 = 5
    var_9 = [var_0, var_7, var_8]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 6/9 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 6/13 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 6/9 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 8/16 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_nested_mixed. Retrieved 9/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_0, var_3, var_4]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = '_fields'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 12/15 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 8/17 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_mixed_nested_structure. Retrieved 17/20 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 3/6 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 12/20 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 7/11 statements.
# Partially parsed test_map_structure_zip_with_multiple_scalars. Retrieved 5/8 statements.
# Partially parsed test_map_structure_zip_preserves_list_structure. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = [var_1, var_3]
    var_8 = 5
    var_9 = [var_4, var_8]
    var_10 = [var_7, var_9]
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = (var_1, var_3)
    var_8 = 5
    var_9 = (var_4, var_8)
    var_10 = (var_7, var_9)
    var_11 = [var_6, var_10]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 2
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 5
    var_10 = 6
    var_11 = [var_9, var_10]
    var_12 = 7
    var_13 = 8
    var_14 = (var_12, var_13)
    var_15 = {var_0: var_11, var_1: var_14}
    var_16 = [var_8, var_15]

def test_case_0():
    var_0 = 5
    var_1 = 3
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'hello'
    var_1 = 'world'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0, var_1}
    var_3 = 3
    var_4 = 4
    var_5 = {var_3, var_4}
    var_6 = [var_2, var_5]
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Structures cannot contain `set`'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_false. Retrieved 14/37 statements.


import flutes.structure as module_0

def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = lambda x, y: x + y
    var_6 = 1
    var_7 = 2
    var_8 = [var_6, var_7]
    var_9 = 3
    var_10 = 4
    var_11 = [var_9, var_10]
    var_12 = [var_8, var_11]
    var_13 = module_0.map_structure_zip(var_5, var_12)
    var_14 = bool(var_13 == [4, 6])
    assert var_14 is True
    var_15 = '__wrapped__'
    var_16 = False



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_preserves_dict_type. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_tuple. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = {var_0: var_4, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 2
    var_3 = (var_1, var_2)
    var_4 = {var_0: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 4
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = ()



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_nested_tuple. Retrieved 8/11 statements.
# Partially parsed test_map_structure_with_namedtuple. Retrieved 7/15 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/9 statements.
# Partially parsed test_map_structure_with_nested_dict. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/8 statements.
# Partially parsed test_map_structure_with_mixed_nested_structure. Retrieved 9/12 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_list. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_empty_dict. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_ordereddict. Retrieved 11/18 statements.
# Partially parsed test_map_structure_with_complex_nested_structure. Retrieved 22/25 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_1, var_2]
    var_4 = 4
    var_5 = [var_4]
    var_6 = [var_5]
    var_7 = [var_0, var_3, var_6]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_1, var_2)
    var_4 = 4
    var_5 = (var_4,)
    var_6 = (var_5,)
    var_7 = (var_0, var_3, var_6)

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 'c'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = (var_5, var_6)
    var_8 = {var_0: var_4, var_1: var_7}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = (var_0, var_4)
    var_8 = 3
    var_9 = (var_3, var_8)
    var_10 = [var_7, var_9]

def test_case_0():
    var_0 = 'list'
    var_1 = 'tuple'
    var_2 = 'dict'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_3, var_4, var_7]
    var_9 = 5
    var_10 = 6
    var_11 = (var_9, var_10)
    var_12 = 'nested'
    var_13 = 7
    var_14 = {var_12: var_13}
    var_15 = {var_0: var_8, var_1: var_11, var_2: var_14}
    var_16 = [var_6, var_9]
    var_17 = [var_4, var_5, var_16]
    var_18 = (var_10, var_13)
    var_19 = 8
    var_20 = {var_12: var_19}
    var_21 = {var_0: var_17, var_1: var_18, var_2: var_20}



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_map_structure_zip_with_lists. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_lists. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_tuples. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_tuples. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_namedtuple. Retrieved 9/18 statements.
# Partially parsed test_map_structure_zip_with_dicts. Retrieved 9/12 statements.
# Partially parsed test_map_structure_zip_with_nested_dicts. Retrieved 15/18 statements.
# Partially parsed test_map_structure_zip_with_mixed_nested_structures. Retrieved 13/16 statements.
# Partially parsed test_map_structure_zip_with_scalars. Retrieved 4/10 statements.
# Partially parsed test_map_structure_zip_with_strings. Retrieved 4/7 statements.
# Partially parsed test_map_structure_zip_with_ordered_dict. Retrieved 16/25 statements.
# Partially parsed test_map_structure_zip_with_multiple_arguments. Retrieved 10/13 statements.
# Partially parsed test_map_structure_zip_with_set_raises_error. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 5
    var_8 = 6
    var_9 = [var_7, var_8]
    var_10 = 7
    var_11 = 8
    var_12 = [var_10, var_11]
    var_13 = [var_9, var_12]
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = (var_4, var_5, var_6)
    var_8 = [var_3, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = 5
    var_8 = 6
    var_9 = (var_7, var_8)
    var_10 = 7
    var_11 = 8
    var_12 = (var_10, var_11)
    var_13 = (var_9, var_12)
    var_14 = [var_6, var_13]

def test_case_0():
    var_0 = 'Point'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 6

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = 3
    var_10 = {var_2: var_9}
    var_11 = 4
    var_12 = {var_5: var_11}
    var_13 = {var_0: var_10, var_1: var_12}
    var_14 = [var_8, var_13]

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = {var_0: var_7}
    var_9 = 4
    var_10 = {var_3: var_9}
    var_11 = [var_8, var_10]
    var_12 = [var_6, var_11]

def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = 4
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'y'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 3
    var_8 = (var_0, var_7)
    var_9 = 4
    var_10 = (var_3, var_9)
    var_11 = [var_8, var_10]
    var_12 = (var_0, var_9)
    var_13 = 6
    var_14 = (var_3, var_13)
    var_15 = [var_12, var_14]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = 5
    var_7 = 6
    var_8 = [var_6, var_7]
    var_9 = [var_2, var_5, var_8]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'set'
    var_7 = bool('set' in str(e).lower())
    assert var_7 is True



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_map_structure_zip_predicate_line_1_evaluates_to_false. Retrieved 10/31 statements.


def test_case_0():
    var_0 = 'R'
    var_1 = []
    var_2 = 'T'
    var_3 = []
    var_4 = '__no_map__'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_8]
    var_10 = var_8.__class__
    var_11 = hasattr(var_8, var_4)



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_map_structure_with_list. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_nested_list. Retrieved 7/10 statements.
# Partially parsed test_map_structure_with_tuple. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_dict. Retrieved 5/8 statements.
# Partially parsed test_map_structure_with_set. Retrieved 4/7 statements.
# Partially parsed test_map_structure_with_scalar. Retrieved 1/4 statements.
# Partially parsed test_map_structure_with_string. Retrieved 1/4 statements.
# Partially parsed test_map_structure_no_type_check_decorator_exists. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3
    var_4 = 4
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = '__wrapped__'



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_map_structure_predicate_line_1_evaluates_to_false. Retrieved 12/29 statements.


def test_case_0():
    var_0 = 'T'
    var_1 = []
    var_2 = 'R'
    var_3 = []
    var_4 = ()
    var_5 = '__no_map__'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = var_9.__class__
    var_11 = var_10 in var_4
    var_12 = hasattr(var_9, var_5)
    var_13 = var_11 or var_12
    assert var_13 is False



