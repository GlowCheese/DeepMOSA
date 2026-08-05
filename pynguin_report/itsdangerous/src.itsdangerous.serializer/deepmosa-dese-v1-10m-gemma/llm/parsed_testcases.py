####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_serializer_kwargs. Retrieved 4/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'extra'
    var_2 = 'arg'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'extra'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_iter_unsigners_default_behavior. Retrieved 9/24 statements.
# Partially parsed test_iter_unsigners_with_fallback_dict. Retrieved 6/19 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 6/23 statements.
# Partially parsed test_iter_unsigners_key_rotation. Retrieved 8/21 statements.
# Partially parsed test_iter_unsigners_with_custom_salt. Retrieved 3/8 statements.
# Partially parsed test_iter_unsigners_fallback_signer_class. Retrieved 3/19 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = b'data'
    var_5 = lambda x: var_4
    var_6 = lambda x: x
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = 0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'extra'
    var_3 = 'arg'
    var_4 = {var_2: var_3}
    var_5 = [var_4]

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'extra'
    var_3 = 'arg'
    var_4 = {var_2: var_3}
    var_5 = 1

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = b'salt'
    var_4 = 'extra'
    var_5 = 'arg'
    var_6 = {var_4: var_5}
    var_7 = [var_6]

def test_case_0():
    var_0 = b'secret'
    var_1 = b'default_salt'
    var_2 = b'new_salt'

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 1



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pdataserializer_dumps_success. Retrieved 6/10 statements.
# Partially parsed test_pdataserializer_dumps_type_consistency. Retrieved 2/7 statements.


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "serialized_{'key': 'value'}"
    var_4 = 123
    var_5 = module_0.dumps(var_4)
    assert var_5 == 'serialized_123'

import json as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.dumps(var_0)
    assert var_1 == 'True'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret_bytes'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'custom_salt'
    var_2 = 'indent'
    var_3 = 4
    var_4 = {var_2: var_3}
    var_5 = 'some_param'
    var_6 = True
    var_7 = {var_5: var_6}
    var_8 = 'extra'
    var_9 = 'dict'
    var_10 = {var_8: var_9}
    var_11 = [var_10]
    var_12 = module_0.Serializer(var_0, var_1, serializer_kwargs=var_4, signer_kwargs=var_7, fallback_signers=var_11)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #5
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serializer_constructor_with_all_args. Retrieved 12/22 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

def test_case_0():
    var_0 = 'signer_kwargs'
    var_1 = 'extra'
    var_2 = 'arg'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_4]
    var_6 = b'key1'
    var_7 = b'salt'
    var_8 = 'indent'
    var_9 = 4
    var_10 = {var_8: var_9}
    var_11 = {var_1: var_2}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key'
    var_1 = b'custom_salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = None
    var_2 = module_0.Serializer(var_0, fallback_signers=var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serializer_constructor_custom_fallback_signers. Retrieved 7/12 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'some'
    var_2 = 'arg'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'extra'
    var_1 = 'arg'
    var_2 = {var_0: var_1}
    var_3 = 'key'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_signer_and_fallback. Retrieved 4/16 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'extra'
    var_1 = 'arg'
    var_2 = {var_0: var_1}
    var_3 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1, fallback_signers=var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_init_with_provided_serializer. Retrieved 1/8 statements.


def test_case_0():
    var_0 = b'secret'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = b'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'signer_kwargs'
    var_1 = 'extra'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = [var_4]
    var_6 = 'secret'
    var_7 = module_0.Serializer(var_6, fallback_signers=var_5)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_serializer_constructor_fallback_signers. Retrieved 7/12 statements.
# Partially parsed test_serializer_constructor_with_custom_signer. Retrieved 1/6 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

def test_case_0():
    var_0 = 'some'
    var_1 = 'dict'
    var_2 = {var_0: var_1}
    var_3 = 'key'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = 'secret'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #12
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_payload_success_with_text_serializer. Retrieved 2/12 statements.
# Partially parsed test_load_payload_success_with_bytes_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 4/15 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 2/17 statements.
# Partially parsed test_load_payload_handles_utf8_decoding_correctly. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"a": 1}'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'some_payload'
    var_2 = 'BadPayload was not raised'
    var_3 = AssertionError(var_2)

def test_case_0():
    var_0 = 'secret'
    var_1 = b'some_data'

def test_case_0():
    var_0 = 'secret'
    var_1 = '{"emoji": "🚀"}'
    var_2 = 'utf-8'



# Parsed testcases at query #14
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = []
    var_3 = module_0.Serializer(var_0, var_1, fallback_signers=var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pdataserializer_dumps_returns_serialized_payload. Retrieved 5/9 statements.
# Partially parsed test_pdataserializer_dumps_handles_primitive_types. Retrieved 6/10 statements.


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = "serialized_{'key': 'value'}"
    var_4 = module_0.dumps(var_2)

import json as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.dumps(var_0)
    assert var_1 == '123'
    var_2 = True
    var_3 = module_0.dumps(var_2)
    assert var_3 == 'True'
    var_4 = None
    var_5 = module_0.dumps(var_4)
    assert var_5 == 'None'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_init_with_serializer_kwargs. Retrieved 4/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'sha256'
    var_2 = {var_0: var_1}
    var_3 = b'secret'
    var_4 = module_0.Serializer(var_3, signer_kwargs=var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'digest_method'
    var_1 = 'sha512'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = b'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

def test_case_0():
    var_0 = b'secret'

def test_case_0():
    var_0 = 'indent'
    var_1 = 4
    var_2 = {var_0: var_1}
    var_3 = b'secret'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_iter_unsigners_default_behavior. Retrieved 6/18 statements.
# Partially parsed test_iter_unsigners_with_fallback_dict. Retrieved 6/16 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 5/15 statements.
# Partially parsed test_iter_unsigners_key_rotation. Retrieved 6/16 statements.
# Partially parsed test_iter_unsigners_custom_salt. Retrieved 3/10 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = b'{"a": 1}'
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'some_kwarg'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = [var_4]

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'kwarg'
    var_3 = 'val'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = b'old_key'
    var_1 = b'new_key'
    var_2 = [var_0, var_1]
    var_3 = b'salt'
    var_4 = {}
    var_5 = 0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'original_salt'
    var_2 = b'custom_salt'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_iter_unsigners_tuple_fallback. Retrieved 7/19 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'extra'
    var_3 = 'arg'
    var_4 = {var_2: var_3}
    var_5 = 0
    var_6 = 1



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 4/9 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret_bytes'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'extra'
    var_2 = 'arg'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'extra'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'secret'

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_payload_with_default_serializer_and_bytes_payload. Retrieved 7/9 statements.
# Partially parsed test_load_payload_with_text_serializer_and_bytes_payload. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 3/10 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 2/12 statements.


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_1.dumps(var_4)
    var_6 = 'utf-8'

def test_case_0():
    var_0 = b'secret'
    var_1 = b'{"key": "value"}'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'some_payload'

def test_case_0():
    var_0 = b'secret'
    var_1 = b'corrupted'



# Parsed testcases at query #21
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = []
    var_3 = None
    var_4 = None
    var_5 = None
    var_6 = None
    var_7 = module_0.Serializer(var_0, var_1, var_3, var_5, var_4, var_6, var_2)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pdataserializer_loads_returns_expected_value. Retrieved 2/8 statements.
# Partially parsed test_pdataserializer_loads_with_different_payload. Retrieved 2/8 statements.


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)

import json as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.loads(var_0)
    assert var_1 == 123



# Parsed testcases at query #23
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'test_salt'
    var_1 = b'secret'
    var_2 = module_0.Serializer(var_1, var_0)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_payload_raises_bad_payload_on_serializer_exception. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'some_data'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_serializer_dumps_returns_signed_bytes. Retrieved 8/11 statements.
# Partially parsed test_serializer_dumps_with_text_serializer. Retrieved 6/16 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5, var_1)
    var_7 = var_2.loads(var_6, var_1)

import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'original_salt'
    var_2 = b'alternative_salt'
    var_3 = module_0.Serializer(var_0, var_1)
    var_4 = var_3
    var_5 = 'some_data'
    var_6 = module_1.dumps(var_5)
    var_7 = module_1.loads(var_6)

import json as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)
    var_5 = module_0.loads(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)
    var_5 = 'a'
    var_6 = 1
    var_7 = {var_5: var_6}
    var_8 = var_4.dumps(var_7)
    var_9 = var_4.loads(var_8)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_iter_unsigners_tuple_fallback. Retrieved 9/20 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'signer_kwargs'
    var_3 = 'extra'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 0
    var_8 = 1



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_payload_exception_raises_bad_payload. Retrieved 4/19 statements.
# Partially parsed test_load_payload_with_override_serializer_exception. Retrieved 2/10 statements.


def test_case_0():
    var_0 = b'data'
    var_1 = 'data'
    var_2 = b'secret'
    var_3 = b'some_payload'

def test_case_0():
    var_0 = b'secret'
    var_1 = b'payload'



# Parsed testcases at query #28
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'extra_arg'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = module_0.Serializer(var_0, var_1, fallback_signers=var_5)
    var_7 = var_6.iter_unsigners()
    var_8 = list(var_7)
    var_9 = len(var_8)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_bytes. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer_text. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'some'
    var_2 = 'arg'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'salt'
    var_1 = 'other'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_payload_with_default_serializer. Retrieved 7/16 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 3/11 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_error. Retrieved 5/19 statements.
# Partially parsed test_load_payload_text_serializer_logic. Retrieved 2/11 statements.


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_1.dumps(var_4)
    var_6 = 'utf-8'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'some_payload'

def test_case_0():
    var_0 = b'secret'
    var_1 = b'any_payload'
    var_2 = 'Should have raised BadPayload'
    var_3 = AssertionError(var_2)
    var_4 = str(var_3)

def test_case_0():
    var_0 = b'secret'
    var_1 = b'{"a": 1}'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_payload_success_with_text_serializer. Retrieved 2/12 statements.
# Partially parsed test_load_payload_success_with_binary_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_error. Retrieved 4/17 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 2/17 statements.
# Partially parsed test_load_payload_decodes_utf8_for_text_serializer. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"a": 1}'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'{invalid json}'
    var_2 = 'BadPayload was not raised'
    var_3 = AssertionError(var_2)

def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"a": 1}'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"msg": "hello"}'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_tuple_fallback. Retrieved 4/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'salt'
    var_1 = 'other_salt'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

def test_case_0():
    var_0 = 'extra'
    var_1 = 'arg'
    var_2 = {var_0: var_1}
    var_3 = 'secret'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_iter_unsigners_basic_functionality. Retrieved 8/25 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 5/26 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'salt1'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = var_2.iter_unsigners()
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'salt1'
    var_2 = 'extra'
    var_3 = 'arg'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = module_0.Serializer(var_0, var_1, fallback_signers=var_5)
    var_7 = var_6.iter_unsigners()
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 2

def test_case_0():
    var_0 = b'key1'
    var_1 = b'salt1'
    var_2 = 'extra'
    var_3 = 'arg'
    var_4 = {var_2: var_3}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old_key'
    var_1 = b'new_key'
    var_2 = [var_0, var_1]
    var_3 = b'salt1'
    var_4 = module_0.Serializer(var_2, var_3)
    var_5 = var_4.iter_unsigners()
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 1

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'key1'
    var_1 = b'original_salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = b'override_salt'
    var_4 = var_2.iter_unsigners(var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serializer_dumps_returns_signed_bytes. Retrieved 9/15 statements.
# Partially parsed test_serializer_dumps_with_text_serializer_returns_str. Retrieved 5/15 statements.
# Partially parsed test_serializer_dumps_uses_serializer_kwargs. Retrieved 8/12 statements.


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = 'foo'
    var_4 = 'bar'
    var_5 = {var_3: var_4}
    var_6 = var_2.dumps(var_5)
    var_7 = module_1.dumps(var_5)
    var_8 = 'utf-8'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'foo'
    var_3 = 'bar'
    var_4 = {var_2: var_3}
    var_5 = b'custom_salt'
    var_6 = var_1.dumps(var_4)
    var_7 = var_1.dumps(var_4, var_5)

import json as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = {var_1: var_2}
    var_4 = module_0.dumps(var_3)

import json as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = 'foo'
    var_5 = 'bar'
    var_6 = {var_4: var_5}
    var_7 = module_0.dumps(var_6)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_iter_unsigners_skips_fallback_loop_when_no_fallbacks. Retrieved 2/12 statements.


def test_case_0():
    var_0 = b'key1'
    var_1 = []



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_init_salt_is_not_none. Retrieved 2/6 statements.


def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'



# Parsed testcases at query #9
#--------------------------




import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_payload_success_with_default_serializer. Retrieved 7/11 statements.
# Partially parsed test_load_payload_success_with_custom_text_serializer. Retrieved 2/11 statements.
# Partially parsed test_load_payload_success_with_custom_bytes_serializer. Retrieved 2/10 statements.
# Partially parsed test_load_payload_with_override_serializer. Retrieved 3/11 statements.


import src.itsdangerous.serializer as module_0
import json as module_1

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_1.dumps(var_4)
    var_6 = 'utf-8'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"a": 1}'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'hello'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"incomplete_json'
    var_3 = var_1.load_payload(var_2)
    var_4 = 'BadPayload was not raised'
    var_5 = AssertionError(var_4)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'any_data'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_serializer_init_with_custom_serializer. Retrieved 1/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = 'some'
    var_2 = 'arg'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'extra'
    var_1 = 'arg'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = b'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

def test_case_0():
    var_0 = b'secret'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer_and_kwargs. Retrieved 4/12 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'some'
    var_2 = 'arg'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'some'
    var_1 = 'dict'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

def test_case_0():
    var_0 = 'secret'
    var_1 = 'foo'
    var_2 = 'bar'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'secret'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/10 statements.
# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 7/15 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = 'salt'
    var_4 = module_0.Serializer(var_2, var_3)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = 'digest_method'
    var_5 = 'sha256'
    var_6 = {var_4: var_5}
    var_7 = module_0.Serializer(var_0, serializer_kwargs=var_3, signer_kwargs=var_6)

def test_case_0():
    var_0 = 'salt'
    var_1 = 'alt_salt'
    var_2 = {var_0: var_1}
    var_3 = 'custom'
    var_4 = True
    var_5 = {var_3: var_4}
    var_6 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1, fallback_signers=var_1)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_serializer_constructor_with_tuple_fallback. Retrieved 4/8 statements.
# Partially parsed test_serializer_constructor_with_bytes_serializer. Retrieved 1/4 statements.
# Partially parsed test_serializer_constructor_with_text_serializer. Retrieved 1/5 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret_bytes'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'some'
    var_2 = 'arg'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'some_param'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

def test_case_0():
    var_0 = 'param'
    var_1 = 'val'
    var_2 = {var_0: var_1}
    var_3 = 'secret'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pdataserializer_loads_returns_expected_value. Retrieved 2/8 statements.
# Partially parsed test_pdataserializer_loads_with_integer_payload. Retrieved 2/8 statements.
# Partially parsed test_pdataserializer_loads_raises_error_on_invalid_input. Retrieved 2/9 statements.


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)

import json as module_0

def test_case_0():
    var_0 = '123'
    var_1 = module_0.loads(var_0)
    assert var_1 == 123

import json as module_0

def test_case_0():
    var_0 = 'not_a_number'
    var_1 = module_0.loads(var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/10 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 'new'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'salt'
    var_2 = 'extra'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'salt'
    var_1 = 'new_salt'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'secret'
    var_5 = module_0.Serializer(var_4, fallback_signers=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_payload_with_override_serializer. Retrieved 3/16 statements.
# Partially parsed test_load_payload_with_text_serializer_logic. Retrieved 2/11 statements.
# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 3/17 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)
    var_3 = b'{"key": "value"}'
    var_4 = var_2.load_payload(var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)
    var_2 = b'{"a": 1}'

def test_case_0():
    var_0 = b'secret'
    var_1 = b'{"status": "ok"}'

def test_case_0():
    var_0 = b'secret'
    var_1 = b'some_data'
    var_2 = 0



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serializer_init_with_serializer_provided. Retrieved 1/9 statements.


def test_case_0():
    var_0 = b'secret'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

def test_case_0():
    var_0 = 'secret'

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, serializer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'some_arg'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'some_arg'
    var_2 = 'val'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pdataserializer_loads_returns_correct_type. Retrieved 2/9 statements.
# Partially parsed test_pdataserializer_loads_with_different_payload_type. Retrieved 2/8 statements.


import json as module_0

def test_case_0():
    var_0 = '{"key": "value"}'
    var_1 = module_0.loads(var_0)

import json as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.loads(var_0)
    assert var_1 == 11



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_iter_unsigners_default_behavior. Retrieved 13/19 statements.
# Partially parsed test_iter_unsigners_with_fallback_tuple. Retrieved 15/23 statements.
# Partially parsed test_iter_unsigners_with_key_rotation. Retrieved 13/21 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'dumps'
    var_3 = 'loads'
    assert var_3 == 1
    var_4 = lambda x: str(x).encode()
    var_5 = lambda x: x.decode()
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = True
    var_8 = module_0.Serializer(var_0, var_1, var_6)
    var_9 = var_8.iter_unsigners()
    var_10 = list(var_9)
    var_11 = 0
    var_12 = var_10[var_11]

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda x: str(x).encode()
    var_5 = lambda x: x.decode()
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'extra'
    var_8 = 'val'
    var_9 = {var_7: var_8}
    var_10 = [var_9]
    var_11 = module_0.Serializer(var_0, var_1, var_6, fallback_signers=var_10)
    var_12 = var_11.iter_unsigners()
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 2

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda x: str(x).encode()
    var_5 = lambda x: x.decode()
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'extra'
    var_8 = 'val'
    var_9 = {var_7: var_8}
    var_10 = [var_5]
    var_11 = module_0.Serializer(var_0, var_1, var_6, fallback_signers=var_10)
    var_12 = var_11.iter_unsigners()
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 2

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'old'
    var_1 = b'new'
    var_2 = [var_0, var_1]
    var_3 = b'salt'
    var_4 = 'dumps'
    var_5 = 'loads'
    var_6 = lambda x: str(x).encode()
    var_7 = lambda x: x.decode()
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.Serializer(var_2, var_3, var_8, fallback_signers=var_0)
    var_10 = var_9.iter_unsigners()
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 3

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'original_salt'
    var_2 = 'dumps'
    var_3 = 'loads'
    var_4 = lambda x: str(x).encode()
    var_5 = lambda x: x.decode()
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.Serializer(var_0, var_1, var_6)
    var_8 = b'new_salt'
    var_9 = var_7.iter_unsigners(var_8)
    var_10 = list(var_9)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_dumps_returns_serialized_payload. Retrieved 4/8 statements.
# Partially parsed test_dumps_handles_primitive_types. Retrieved 4/8 statements.
# Partially parsed test_dumps_with_complex_object. Retrieved 9/13 statements.


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "serialized_{'key': 'value'}"

import json as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.dumps(var_0)
    assert var_1 == '123'
    var_2 = True
    var_3 = module_0.dumps(var_2)
    assert var_3 == 'True'

import json as module_0

def test_case_0():
    var_0 = 'id'
    var_1 = 'data'
    var_2 = 42
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.dumps(var_7)
    assert var_8 == 42



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_payload_raises_bad_payload_on_exception. Retrieved 2/13 statements.
# Partially parsed test_load_payload_success_path. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'secret'
    var_1 = b'some_data'

def test_case_0():
    var_0 = 'secret'
    var_1 = b'{"key": "value"}'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_serializer_constructor_with_fallback_signers. Retrieved 7/12 statements.
# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/10 statements.
# Partially parsed test_serializer_constructor_with_serializer_kwargs. Retrieved 4/7 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'mysalt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = None
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'some_arg'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.Serializer(var_0, signer_kwargs=var_3)

def test_case_0():
    var_0 = 'extra'
    var_1 = 'kwargs'
    var_2 = {var_0: var_1}
    var_3 = 'other'
    var_4 = 'arg'
    var_5 = {var_3: var_4}
    var_6 = 'secret'

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'indent'
    var_2 = 4
    var_3 = {var_1: var_2}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_serializer_constructor_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serializer_constructor_with_custom_signer_and_kwargs. Retrieved 4/16 statements.


import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = module_0.Serializer(var_0)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = b'secret'
    var_1 = b'salt'
    var_2 = module_0.Serializer(var_0, var_1)

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = b'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Serializer(var_2)

def test_case_0():
    var_0 = 'secret'

def test_case_0():
    var_0 = 'secret'
    var_1 = 'extra'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import src.itsdangerous.serializer as module_0

def test_case_0():
    var_0 = 'secret'
    var_1 = 'extra'
    var_2 = 'fallback'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = module_0.Serializer(var_0, fallback_signers=var_4)
    var_6 = var_5.iter_unsigners()
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_5.iter_unsigners()
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pdataserializer_dumps_returns_payload. Retrieved 4/8 statements.
# Partially parsed test_pdataserializer_dumps_handles_primitive_types. Retrieved 4/8 statements.


import json as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.dumps(var_2)
    assert var_3 == "serialized_{'key': 'value'}"

import json as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.dumps(var_0)
    assert var_1 == '123'
    var_2 = True
    var_3 = module_0.dumps(var_2)
    assert var_3 == 'True'



