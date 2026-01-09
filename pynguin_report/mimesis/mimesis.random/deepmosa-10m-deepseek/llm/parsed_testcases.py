####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generate_string_by_mask_default_mask. Retrieved 9/13 statements.
# Partially parsed test_generate_string_by_mask_custom_mask. Retrieved 19/29 statements.
# Partially parsed test_generate_string_by_mask_custom_placeholders. Retrieved 16/22 statements.
# Partially parsed test_generate_string_by_mask_literal_characters. Retrieved 11/16 statements.
# Partially parsed test_generate_string_by_mask_only_placeholders. Retrieved 4/6 statements.
# Partially parsed test_generate_string_by_mask_mixed_placeholders_and_literals. Retrieved 9/12 statements.


import mimesis.random as module_0


def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = var_1[var_3]
    var_6 = 1
    var_7 = 4
    var_8 = var_1[var_6:var_7]


def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@###-@@'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 8
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = var_2[var_4]
    var_7 = 1
    var_8 = var_2[var_7]
    var_9 = var_2[var_7]
    var_10 = 2
    var_11 = 5
    var_12 = var_2[var_10:var_11]
    var_13 = var_2[5]
    assert var_13 == '-'
    var_14 = 6
    var_15 = var_2[var_14]
    var_16 = var_2[var_14]
    var_17 = 7
    var_18 = var_2[var_17]
    var_19 = var_2[var_17]


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'AA99'
    var_2 = 'A'
    var_3 = '9'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = var_4[var_6]
    var_9 = 1
    var_10 = var_4[var_9]
    var_11 = var_4[var_9]
    var_12 = 2
    var_13 = var_4[var_12]
    var_14 = 3
    var_15 = var_4[var_14]


def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@##'
    var_2 = '@'
    var_3 = var_0.generate_string_by_mask(var_1, var_2, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'same placeholder'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'CODE-@##'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = 'CODE-'
    var_4 = 5
    var_5 = var_2[var_4]
    var_6 = var_2[var_4]
    var_7 = 6
    var_8 = var_2[var_7]
    var_9 = 7
    var_10 = var_2[var_9]


def test_case_0():
    var_0 = module_0.Random()
    var_1 = ''
    var_2 = var_0.generate_string_by_mask(var_1)
    assert var_2 == ''


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'FIXED'
    var_2 = var_0.generate_string_by_mask(var_1)
    assert var_2 == 'FIXED'


def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@@@@'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 5


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'X@Y#Z'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = var_2[0]
    assert var_4 == 'X'
    var_5 = 1
    var_6 = var_2[var_5]
    var_7 = var_2[var_5]
    var_8 = var_2[2]
    assert var_8 == 'Y'
    var_9 = 3
    var_10 = var_2[var_9]
    var_11 = var_2[4]
    assert var_11 == 'Z'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_inherits_from_base_random. Retrieved 1/3 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'generate_string_by_mask'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'uniform'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randbytes'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'weighted_choice'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'choice_enum_item'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'choices'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'choice'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_random_constructor_default. Retrieved 1/2 statements.
# Partially parsed test_random_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_random_constructor_inherits_from_random. Retrieved 1/3 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)


def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_inherits_from_base_random. Retrieved 1/3 statements.
# Partially parsed test_constructor_seed_produces_reproducible_random. Retrieved 3/5 statements.
# Partially parsed test_constructor_no_arguments. Retrieved 1/3 statements.
# Partially parsed test_constructor_with_int_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_float_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_str_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_bytes_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_with_bytearray_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_instance_has_randints_method. Retrieved 2/5 statements.
# Partially parsed test_constructor_instance_has_generate_string_by_mask_method. Retrieved 2/3 statements.
# Partially parsed test_constructor_instance_has_uniform_method. Retrieved 4/5 statements.
# Partially parsed test_constructor_instance_has_randbytes_method. Retrieved 2/3 statements.
# Partially parsed test_constructor_instance_has_choice_enum_item_method. Retrieved 4/7 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = 12345
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0.0


def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = 3.14
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = 'seed'
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = b'bytes_seed'
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = b'bytearray_seed'
    var_1 = bytearray(var_0)
    var_2 = module_0.Random(var_1)


def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints()


def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1.0
    var_2 = 10.0
    var_3 = var_0.uniform(var_1, var_2)


def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes()


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 0.5
    var_4 = {var_1: var_3, var_2: var_3}
    var_5 = var_0.weighted_choice(var_4)
    var_6 = bool(var_5 in var_4)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 2
    var_3 = 3



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 1/5 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 1/3 statements.
# Partially parsed test_constructor_inherits_random_methods. Retrieved 3/4 statements.
# Partially parsed test_constructor_state_initialization. Retrieved 2/7 statements.



def test_case_0():
    var_0 = module_0.Random()

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = None


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 10
    var_3 = 1


def test_case_0():
    var_0 = 123
    var_1 = module_0.Random()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_accepts_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_without_seed. Retrieved 1/2 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 1/5 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 1/3 statements.
# Partially parsed test_constructor_inherits_random_methods. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.Random()

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = None


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 10
    var_3 = 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_constructor_initializes_without_seed. Retrieved 1/2 statements.
# Partially parsed test_constructor_initializes_with_seed. Retrieved 1/3 statements.
# Partially parsed test_constructor_initializes_with_none_seed. Retrieved 1/3 statements.
# Partially parsed test_constructor_initializes_with_int_seed. Retrieved 1/3 statements.
# Partially parsed test_constructor_initializes_with_float_seed. Retrieved 1/3 statements.
# Partially parsed test_constructor_initializes_with_str_seed. Retrieved 1/3 statements.
# Partially parsed test_constructor_initializes_with_bytes_seed. Retrieved 1/3 statements.
# Partially parsed test_constructor_initializes_with_bytearray_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_initializes_with_memoryview_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_initializes_with_version_parameter. Retrieved 1/3 statements.



def test_case_0():
    var_0 = module_0.Random()

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 12345

def test_case_0():
    var_0 = 3.14159

def test_case_0():
    var_0 = 'test_seed'

def test_case_0():
    var_0 = b'bytes_seed'

def test_case_0():
    var_0 = b'bytearray_seed'
    var_1 = bytearray(var_0)

def test_case_0():
    var_0 = b'memoryview_seed'
    var_1 = memoryview(var_0)

def test_case_0():
    var_0 = 2



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_inherits_from_base_random. Retrieved 1/3 statements.
# Partially parsed test_constructor_supports_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_without_seed_produces_different_instances. Retrieved 2/4 statements.
# Partially parsed test_constructor_accepts_none_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_accepts_int_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_accepts_str_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_accepts_bytes_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_accepts_bytearray_seed. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = 12345
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()


def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = 'seed'
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = b'seed'
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = b'seed'
    var_1 = bytearray(var_0)
    var_2 = module_0.Random(var_1)


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'generate_string_by_mask'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'uniform'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'randbytes'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'weighted_choice'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'choice_enum_item'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_inherits_from_base_random. Retrieved 1/3 statements.
# Partially parsed test_constructor_accepts_int_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_accepts_none_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_accepts_float_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_accepts_str_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_accepts_bytes_seed. Retrieved 2/4 statements.
# Partially parsed test_constructor_accepts_bytearray_seed. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()
    var_2 = 5
    var_3 = range(var_2)
    var_4 = [var_0.random() for _ in var_3]
    var_5 = range(var_2)
    var_6 = [var_1.random() for _ in var_5]
    var_7 = bool(var_4 != var_6)
    assert var_7 is True


def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)
    var_3 = 10
    var_4 = range(var_3)
    var_5 = [var_1.random() for _ in var_4]
    var_6 = range(var_3)
    var_7 = [var_2.random() for _ in var_6]
    var_8 = bool(var_5 == var_7)
    assert var_8 is True


def test_case_0():
    var_0 = 123
    var_1 = module_0.Random(var_0)
    var_2 = 456
    var_3 = module_0.Random(var_2)
    var_4 = 10
    var_5 = range(var_4)
    var_6 = [var_1.random() for _ in var_5]
    var_7 = range(var_4)
    var_8 = [var_3.random() for _ in var_7]
    var_9 = bool(var_6 != var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 100
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = 3.14
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = 'seed'
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = b'bytes_seed'
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = b'bytearray_seed'
    var_1 = bytearray(var_0)
    var_2 = module_0.Random(var_1)


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'generate_string_by_mask'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'uniform'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'randbytes'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'weighted_choice'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'choice_enum_item'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.generate_string_by_mask
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = var_0.uniform
    var_8 = callable(var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = var_0.randbytes
    var_11 = callable(var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = var_0.weighted_choice
    var_14 = callable(var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = var_0.choice_enum_item
    var_17 = callable(var_16)
    var_18 = bool(var_17)
    assert var_18 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_accepts_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_same_seed_produces_same_sequence. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_different_seed_produces_different_sequence. Retrieved 4/6 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()
    var_2 = bool(var_0 is not var_1)
    assert var_2 is True


def test_case_0():
    var_0 = 123
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)


def test_case_0():
    var_0 = 1
    var_1 = module_0.Random(var_0)
    var_2 = 2
    var_3 = module_0.Random(var_2)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_inherits_from_random. Retrieved 1/3 statements.
# Partially parsed test_constructor_can_be_seeded. Retrieved 3/5 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.randints
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'generate_string_by_mask'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.generate_string_by_mask
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'uniform'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.uniform
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randbytes'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.randbytes
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'weighted_choice'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.weighted_choice
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'choice_enum_item'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0.choice_enum_item
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = '_generate_string'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = var_0._generate_string
    var_5 = callable(var_4)
    var_6 = bool(var_5)
    assert var_6 is True


def test_case_0():
    var_0 = 12345
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_inherits_from_random. Retrieved 1/3 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.uniform
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randbytes
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.weighted_choice
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True


def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.choice_enum_item
    var_2 = callable(var_1)
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_accepts_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_same_seed_produces_same_sequence. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_different_seed_produces_different_sequence. Retrieved 4/6 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()
    var_2 = bool(var_0 is not var_1)
    assert var_2 is True


def test_case_0():
    var_0 = 123
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)


def test_case_0():
    var_0 = 1
    var_1 = module_0.Random(var_0)
    var_2 = 2
    var_3 = module_0.Random(var_2)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 5/7 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_string_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_float_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_inherits_methods. Retrieved 3/4 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)
    var_3 = 1
    var_4 = 100


def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = 'seed'
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = 3.14
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 10
    var_3 = 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_accepts_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_same_seed_produces_same_random_sequence. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_different_seed_produces_different_random_sequence. Retrieved 4/6 statements.
# Partially parsed test_constructor_inherits_from_random_module. Retrieved 1/3 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()
    var_2 = bool(var_0 is not var_1)
    assert var_2 is True


def test_case_0():
    var_0 = 123
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)


def test_case_0():
    var_0 = 1
    var_1 = module_0.Random(var_0)
    var_2 = 2
    var_3 = module_0.Random(var_2)


def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_inherits_from_base_random. Retrieved 1/3 statements.
# Partially parsed test_constructor_accepts_seed. Retrieved 1/5 statements.
# Partially parsed test_constructor_without_seed_produces_different_values. Retrieved 2/4 statements.



def test_case_0():
    var_0 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()

def test_case_0():
    var_0 = 42


def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()


def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'generate_string_by_mask'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'uniform'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'randbytes'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'weighted_choice'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'choice_enum_item'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = '_generate_string'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



