####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.com'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'
    var_3 = '-test'
    var_4 = module_0.suffix(var_3)
    var_5 = 'word'



# Parsed testcases at query #2
#--------------------------


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'default'
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'value'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Как дела?'
    var_2 = 'Привіт'
    var_3 = 'Як справи?'
    var_4 = 'Сәлем'
    var_5 = 'Қалайсың?'
    var_6 = 123



# Parsed testcases at query #4
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 6
    var_6 = 1
    var_7 = lambda x: x + var_6
    var_8 = module_0.apply_if(var_1, var_3, var_7)
    var_9 = 4
    var_10 = module_0.apply_if(var_1, var_3)
    var_11 = 3
    var_12 = lambda x: x.upper()
    var_13 = module_0.apply_if(var_1, var_12)
    var_14 = 'hello'
    var_15 = lambda x: x.lower()
    var_16 = module_0.apply_if(var_1, var_12, var_15)
    var_17 = 'hi'
    var_18 = 123
    var_19 = module_0.apply_if(var_18, var_12)
    var_20 = 123
    var_21 = module_0.apply_if(var_1, var_20)
    var_22 = 123
    var_23 = module_0.apply_if(var_1, var_12, var_22)
    var_24 = 'All test cases passed'
    var_25 = print(var_24)



# Parsed testcases at query #5
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'
    var_3 = 'test_'
    var_4 = module_0.prefix(var_3)
    var_5 = 'case'
    var_6 = 'abc'
    var_7 = module_0.prefix(var_6)
    var_8 = '123'
    var_9 = ''
    var_10 = module_0.prefix(var_9)
    var_11 = 'hello'
    var_12 = 'pre_'
    var_13 = module_0.prefix(var_12)
    var_14 = 'x'
    var_15 = module_0.prefix(var_14)
    var_16 = 'yz'



# Parsed testcases at query #6
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 3
    var_1 = lambda x: len(x) > var_0
    var_2 = lambda x: x.upper()
    var_3 = module_0.apply_if(var_1, var_2)
    var_4 = 'test'
    var_5 = lambda x: x.lower()
    var_6 = module_0.apply_if(var_1, var_2, var_5)
    var_7 = module_0.apply_if(var_1, var_2)
    var_8 = module_0.apply_if(var_1, var_2)
    var_9 = ''



# Parsed testcases at query #7
#--------------------------


import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'user-'
    var_2 = module_0.prefix(var_1)
    var_3 = 'John Doe'
    var_4 = module_1.Random()
    var_5 = -1
    var_6 = 'TEST'
    var_7 = -1
    var_8 = module_0.pipe()
    var_9 = 'All pipe() tests passed!'
    var_10 = print(var_9)



# Parsed testcases at query #8
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'sensitive_info'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'sensitive_info'

import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'sensitive_info'



# Parsed testcases at query #9
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'



# Parsed testcases at query #10
#--------------------------


import mimesis.keys as module_0
import base64 as module_1

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'hello'
    var_3 = module_1.encode()



# Parsed testcases at query #11
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'
    var_3 = 'https://'
    var_4 = module_0.prefix(var_3)
    var_5 = 'example.com'
    var_6 = 'test'
    var_7 = module_0.prefix(var_6)
    var_8 = 123



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = 'hello'



# Parsed testcases at query #13
#--------------------------


import mimesis.keys as module_0
import mimesis.random as module_1

def test_case_0():
    var_0 = 1
    var_1 = 'TEST_'
    var_2 = module_0.prefix(var_1)
    var_3 = 'hello'
    var_4 = '_end'
    var_5 = module_0.suffix(var_4)
    var_6 = 'HELLO'
    var_7 = 'abc'
    var_8 = module_1.Random()
    var_9 = 'test'
    var_10 = 'TEST'
    var_11 = 'All pipe tests passed!'
    var_12 = print(var_11)



# Parsed testcases at query #14
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'sha256'
    var_1 = module_0.hash_with(var_0)
    var_2 = 'hello'
    var_3 = 'world'
    var_4 = 'md5'
    var_5 = module_0.hash_with(var_4)
    var_6 = 'unsupported_algorithm'
    var_7 = module_0.hash_with(var_6)
    var_8 = 123



# Parsed testcases at query #15
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = '\n    Unit test for function hash_with.\n    '
    var_1 = 'md5'
    var_2 = module_0.hash_with(var_1)
    var_3 = 'hello'
    var_4 = 'sha1'
    var_5 = module_0.hash_with(var_4)
    var_6 = 'world'
    var_7 = 'sha256'
    var_8 = module_0.hash_with(var_7)
    var_9 = 'mimesis'
    var_10 = 'unsupported_algorithm'
    var_11 = module_0.hash_with(var_10)
    var_12 = 'test'
    var_13 = 123



# Parsed testcases at query #16
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.pipe()



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'TEST'
    var_2 = 3
    var_3 = 6
    var_4 = 'All tests for apply_if passed!'
    var_5 = print(var_4)



# Parsed testcases at query #18
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = ' | '
    var_6 = module_0.join(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = '_'
    var_9 = module_0.join(var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = ''
    var_12 = module_0.join(var_11)
    var_13 = [var_1, var_2, var_3]
    var_14 = module_0.join()
    var_15 = 123



# Parsed testcases at query #19
#--------------------------


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test'
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'original'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 'a'



# Parsed testcases at query #21
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = '('
    var_1 = ')'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'
    var_4 = '['
    var_5 = ']'
    var_6 = module_0.wrap(var_4, var_5)
    var_7 = '<'
    var_8 = '>'
    var_9 = module_0.wrap(var_7, var_8)
    var_10 = ''
    var_11 = module_0.wrap(var_10, var_10)
    var_12 = 'a'
    var_13 = 'b'
    var_14 = module_0.wrap(var_12, var_13)



# Parsed testcases at query #22
#--------------------------


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'test_value'
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'original_value'



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'tst'



# Parsed testcases at query #24
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'
    var_3 = 'profile'
    var_4 = 123



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Дякую'
    var_2 = 'Сәлем'
    var_3 = 'Expected ValueError for unsupported locale'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #26
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'secret'
    var_2 = 'CLASSIFIED'
    var_3 = module_0.redact(var_2)
    var_4 = 'top_secret'
    var_5 = 'XXX'
    var_6 = module_0.redact(var_5)
    var_7 = 'anything'



# Parsed testcases at query #27
#--------------------------


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'default_value'
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'test_value'



# Parsed testcases at query #28
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = '[CLASSIFIED]'
    var_1 = module_0.redact(var_0)
    var_2 = 'sensitive_data'
    var_3 = 12345
    var_4 = None



# Parsed testcases at query #29
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.hash_with()
    var_1 = 'test'
    var_2 = 'sha1'
    var_3 = module_0.hash_with(var_2)
    var_4 = 'md5'
    var_5 = module_0.hash_with(var_4)
    var_6 = 'unsupported'
    var_7 = module_0.hash_with(var_6)
    var_8 = 123



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 3
    var_2 = 4



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.io'
    var_1 = module_0.suffix(var_0)
    var_2 = 'ecipe'
    var_3 = '_test'
    var_4 = module_0.suffix(var_3)
    var_5 = 'example'
    var_6 = ''
    var_7 = module_0.suffix(var_6)
    var_8 = 'hello'
    var_9 = '123'
    var_10 = module_0.suffix(var_9)
    var_11 = 'abc'
    var_12 = '!'
    var_13 = module_0.suffix(var_12)
    var_14 = 'world'
    var_15 = 123
    var_16 = module_0.suffix(var_15)
    var_17 = 'test'
    var_18 = '.io'
    var_19 = module_0.suffix(var_18)
    var_20 = 123



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Мир'
    var_2 = 'Привіт'
    var_3 = 'Світ'
    var_4 = 'Сәлем'
    var_5 = 'Әлем'
    var_6 = 123



# Parsed testcases at query #3
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'This is a long string'
    var_4 = 'Short'
    var_5 = 'Exactly10'
    var_6 = ''
    var_7 = 'This is exactly 10'
    var_8 = 5
    var_9 = '!!!'
    var_10 = module_0.truncate(var_8, var_9)
    var_11 = 'Long string'
    var_12 = 2
    var_13 = module_0.truncate(var_12, var_1)
    var_14 = 123
    var_15 = 0
    var_16 = '...'
    var_17 = module_0.truncate(var_15, var_16)
    var_18 = -1
    var_19 = '...'
    var_20 = module_0.truncate(var_18, var_19)



# Parsed testcases at query #4
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = ' | '
    var_6 = module_0.join(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = module_0.join()
    var_9 = []
    var_10 = module_0.join()
    var_11 = 123



# Parsed testcases at query #5
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = ', '
    var_1 = 'apple'
    var_2 = 'banana'
    var_3 = 'cherry'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'apple, banana, cherry'
    var_6 = module_0.join(var_0)
    var_7 = ' | '
    var_8 = 'one'
    var_9 = 'two'
    var_10 = 'three'
    var_11 = [var_8, var_9, var_10]
    var_12 = 'one | two | three'
    var_13 = module_0.join(var_7)
    var_14 = ''
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'c'
    var_18 = [var_15, var_16, var_17]
    var_19 = 'abc'
    var_20 = module_0.join(var_14)
    var_21 = '-'
    var_22 = 'hello'
    var_23 = 'world'
    var_24 = [var_22, var_23]
    var_25 = 'hello-world'
    var_26 = module_0.join(var_21)
    var_27 = ' '
    var_28 = 'foo'
    var_29 = 'bar'
    var_30 = [var_28, var_29]
    var_31 = 'foo bar'
    var_32 = module_0.join(var_27)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'test'
    var_1 = 'two'



# Parsed testcases at query #7
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = ' | '
    var_6 = module_0.join(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = '-'
    var_9 = module_0.join(var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = module_0.join()
    var_12 = 123



# Parsed testcases at query #8
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 3
    var_6 = lambda x: x > var_0
    var_7 = lambda x: x * var_2
    var_8 = module_0.apply_if(var_6, var_7)
    var_9 = -3
    var_10 = lambda x: x > var_0
    var_11 = lambda x: x * var_2
    var_12 = lambda x: x * var_5
    var_13 = module_0.apply_if(var_10, var_11, var_12)
    var_14 = -3



# Parsed testcases at query #9
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 1
    var_6 = lambda x: x > var_0
    var_7 = lambda x: x * var_2
    var_8 = 3
    var_9 = lambda x: x * var_8
    var_10 = module_0.apply_if(var_6, var_7, var_9)
    var_11 = -1
    var_12 = lambda x: x > var_0
    var_13 = lambda x: x * var_2
    var_14 = module_0.apply_if(var_12, var_13)
    var_15 = -1
    var_16 = lambda x: len(x) > var_8
    var_17 = 'test'
    var_18 = lambda x: len(x) > var_8
    var_19 = 'hi'



# Parsed testcases at query #10
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Привіт'
    var_2 = 'Сәлем'
    var_3 = 123
    var_4 = module_0.romanize(var_3)
    var_5 = 123



# Parsed testcases at query #11
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.redact()
    var_1 = 'secret'



# Parsed testcases at query #12
#--------------------------


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'default'
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'original'



# Parsed testcases at query #13
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = '('
    var_1 = ')'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'



# Parsed testcases at query #14
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = '('
    var_1 = ')'
    var_2 = module_0.wrap(var_0, var_1)
    var_3 = 'test'
    var_4 = '['
    var_5 = ']'
    var_6 = module_0.wrap(var_4, var_5)
    var_7 = '<'
    var_8 = '>'
    var_9 = module_0.wrap(var_7, var_8)
    var_10 = ''
    var_11 = module_0.wrap(var_10, var_10)
    var_12 = 'a'
    var_13 = 'b'
    var_14 = module_0.wrap(var_12, var_13)



# Parsed testcases at query #15
#--------------------------


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = 'Test maybe function.'
    var_1 = module_0.Random()
    var_2 = 'test'
    var_3 = 'alternative'
    var_4 = module_1.maybe(var_3)
    var_5 = 1.0
    var_6 = module_1.maybe(var_3, var_5)
    var_7 = 0.0
    var_8 = module_1.maybe(var_3, var_7)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 5



# Parsed testcases at query #17
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = lambda x: x > var_0
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = module_0.apply_if(var_1, var_3)
    var_5 = 6
    var_6 = lambda x: x > var_0
    var_7 = lambda x: x * var_2
    var_8 = 1
    var_9 = lambda x: x + var_8
    var_10 = module_0.apply_if(var_6, var_7, var_9)
    var_11 = 3
    var_12 = lambda x: x > var_0
    var_13 = lambda x: x * var_2
    var_14 = module_0.apply_if(var_12, var_13)
    var_15 = lambda x: len(x) > var_11
    var_16 = lambda x: x.upper()
    var_17 = module_0.apply_if(var_15, var_16)
    var_18 = 'test'
    var_19 = lambda x: len(x) > var_11
    var_20 = lambda x: x.upper()
    var_21 = lambda x: x.lower()
    var_22 = module_0.apply_if(var_19, var_20, var_21)
    var_23 = 'hi'



# Parsed testcases at query #18
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'Test the prefix function.'
    var_1 = 'user_'
    var_2 = module_0.prefix(var_1)
    var_3 = 'name'
    var_4 = 'email'
    var_5 = 123



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'John Doe'
    var_1 = 'Alice Smith'
    var_2 = 'Hello'



# Parsed testcases at query #20
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.join()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = ' | '
    var_6 = module_0.join(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = ''
    var_9 = module_0.join(var_8)
    var_10 = [var_1, var_2, var_3]
    var_11 = ' '
    var_12 = module_0.join(var_11)
    var_13 = [var_1]
    var_14 = module_0.join(var_11)
    var_15 = []



# Parsed testcases at query #21
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.truncate(var_0)
    var_2 = 'abcdef'
    var_3 = 'abc'
    var_4 = 10
    var_5 = '!!!'
    var_6 = module_0.truncate(var_4, var_5)
    var_7 = 'abcdefghijklmn'
    var_8 = 'abcde'
    var_9 = ''
    var_10 = 123
    var_11 = 0
    var_12 = module_0.truncate(var_11)
    var_13 = -1
    var_14 = module_0.truncate(var_13)



# Parsed testcases at query #22
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = '.com'
    var_1 = module_0.suffix(var_0)
    var_2 = 'example'
    var_3 = ''
    var_4 = module_0.suffix(var_3)
    var_5 = '.net'
    var_6 = module_0.suffix(var_5)
    var_7 = '@gmail.com'
    var_8 = module_0.suffix(var_7)
    var_9 = 'user'
    var_10 = '.org'
    var_11 = module_0.suffix(var_10)
    var_12 = 123
    var_13 = 'All suffix tests passed.'
    var_14 = print(var_13)



# Parsed testcases at query #23
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'
    var_2 = '['
    var_3 = ']'
    var_4 = module_0.wrap(var_2, var_3)
    var_5 = module_0.wrap()
    var_6 = 123



# Parsed testcases at query #24
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 3
    var_1 = -3
    var_2 = -3
    var_3 = 'hello'
    var_4 = 'hi'
    var_5 = 4
    var_6 = 9
    var_7 = 8
    var_8 = 0
    var_9 = lambda x: x > var_8
    var_10 = 2
    var_11 = lambda x: x * var_10
    var_12 = module_0.apply_if(var_9, var_11)
    var_13 = lambda x: x > var_8
    var_14 = lambda x: x * var_10
    var_15 = module_0.apply_if(var_13, var_14)
    var_16 = -3
    var_17 = lambda x: x > var_8
    var_18 = lambda x: x * var_10
    var_19 = lambda x: x ** var_10
    var_20 = module_0.apply_if(var_17, var_18, var_19)
    var_21 = -3
    var_22 = None
    var_23 = 5
    var_24 = ''
    var_25 = 1
    var_26 = [var_25, var_10, var_0]
    var_27 = [var_25, var_10]
    var_28 = 'key'
    var_29 = 'value'
    var_30 = {var_28: var_29}
    var_31 = 'other_key'
    var_32 = {var_31: var_29}
    var_33 = 'Alice'
    var_34 = 20
    var_35 = 'Bob'
    var_36 = 16
    var_37 = 'adult'
    var_38 = 'All tests passed!'
    var_39 = print(var_38)



# Parsed testcases at query #25
#--------------------------


import mimesis.random as module_0
import mimesis.keys as module_1

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'default'
    var_2 = 0.5
    var_3 = module_1.maybe(var_1, var_2)
    var_4 = 'original'



# Parsed testcases at query #26
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = '...'
    var_2 = module_0.truncate(var_0, var_1)
    var_3 = 'This is a long sentence'
    var_4 = 'Short'
    var_5 = 'Exactly ten '
    var_6 = ''
    var_7 = 123
    var_8 = -1
    var_9 = module_0.truncate(var_8)



# Parsed testcases at query #27
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'
    var_3 = 'profile'
    var_4 = '123'
    var_5 = 123



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Мир'
    var_2 = 'Привіт'
    var_3 = 'Світ'
    var_4 = 'Сәлем'
    var_5 = 'Әлем'
    var_6 = 123



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'Привет'
    var_1 = 'Мир'
    var_2 = 'Дом'
    var_3 = 123



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test the apply_if function.'
    var_1 = 'test'
    var_2 = 'TEST'



# Parsed testcases at query #31
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 'user_'
    var_1 = module_0.prefix(var_0)
    var_2 = 'order'
    var_3 = 'test_'
    var_4 = module_0.prefix(var_3)
    var_5 = 'case'
    var_6 = ''
    var_7 = module_0.prefix(var_6)
    var_8 = 'empty'



# Parsed testcases at query #32
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.truncate(var_0)
    var_2 = 'This is a very long string'
    var_3 = 5
    var_4 = ''
    var_5 = module_0.truncate(var_3, var_4)
    var_6 = 'Hello, World!'
    var_7 = 20
    var_8 = module_0.truncate(var_7)
    var_9 = 'Short'
    var_10 = '..'
    var_11 = module_0.truncate(var_3, var_10)
    var_12 = -1
    var_13 = module_0.truncate(var_12)
    var_14 = 'Test'
    var_15 = 5
    var_16 = module_0.truncate(var_15)
    var_17 = 123



# Parsed testcases at query #33
#--------------------------


import mimesis.keys as module_0

def test_case_0():
    var_0 = module_0.wrap()
    var_1 = 'test'
    var_2 = '['
    var_3 = ']'
    var_4 = module_0.wrap(var_2, var_3)
    var_5 = module_0.wrap()
    var_6 = ''
    var_7 = module_0.wrap()
    var_8 = 123



