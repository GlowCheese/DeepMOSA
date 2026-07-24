####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 12
    var_7 = 30
    var_8 = 0
    var_9 = 5
    var_10 = module_1.timedelta()
    var_11 = 1
    var_12 = -8
    var_13 = module_1.timedelta()
    var_14 = 20
    var_15 = 15
    var_16 = 45
    var_17 = 123456
    var_18 = 25
    var_19 = '2023-10-27'
    var_20 = var_0.serialize(var_19)



# Parsed testcases at query #2
#--------------------------


import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '550e8400-e29b-41d4-a716-446655440000'
    var_2 = '6ba7b810-9dad-11d1-80b4-00c04fd430c8'
    var_3 = var_0.validate(var_2)
    var_4 = module_1.UUID(var_2)
    var_5 = 'not-a-uuid'
    var_6 = var_0.validate(var_5)
    var_7 = '550e8400-e29b-41d4-a716'
    var_8 = var_0.validate(var_7)
    var_9 = 'zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz'
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #3
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = 12
    var_2 = 30
    var_3 = 45
    var_4 = 123000
    var_5 = None
    var_6 = var_0.serialize(var_5)
    assert var_6 is None
    var_7 = '12:30:45'
    var_8 = var_0.serialize(var_7)
    var_9 = 2023
    var_10 = 1
    var_11 = 12
    var_12 = 30
    var_13 = var_0.serialize(var_4)



# Parsed testcases at query #4
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '8.8.8.8'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '255.255.255.255'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.IPv6Address(var_13)
    var_16 = '::1'
    var_17 = var_0.validate(var_16)
    var_18 = module_1.IPv6Address(var_16)
    var_19 = 'not-an-ip'
    var_20 = var_0.validate(var_19)
    var_21 = '127.0.0.256'
    var_22 = var_0.validate(var_21)
    var_23 = '999.999.999.999'
    var_24 = var_0.validate(var_23)



# Parsed testcases at query #5
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2023
    var_2 = 10
    var_3 = 27



# Parsed testcases at query #6
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-05-20'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 5
    var_5 = 20
    var_6 = '2000-01-01'
    var_7 = var_0.validate(var_6)
    var_8 = 2000
    var_9 = 1
    var_10 = '20-05-2023'
    var_11 = var_0.validate(var_10)
    var_12 = 'not-a-date'
    var_13 = var_0.validate(var_12)
    var_14 = '2023-02-30'
    var_15 = var_0.validate(var_14)
    var_16 = '2023-1-1'
    var_17 = var_0.validate(var_16)



# Parsed testcases at query #7
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = '2023-10-27 15:30:45'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-10-27T15:30:45+02:00'
    var_12 = var_0.validate(var_11)
    var_13 = 2
    var_14 = module_1.timedelta()
    var_15 = '2023-10-27T15:30:45-05:00'
    var_16 = var_0.validate(var_15)
    var_17 = -5
    var_18 = module_1.timedelta()
    var_19 = '2023-10-27T15:30:45.123456'
    var_20 = var_0.validate(var_19)
    var_21 = 123456
    var_22 = '2023-10-27T15:30:45.12'
    var_23 = var_0.validate(var_22)
    var_24 = 120000
    var_25 = '27-10-2023 15:30:45'
    var_26 = var_0.validate(var_25)
    var_27 = '2023/10/27 15:30:45'
    var_28 = var_0.validate(var_27)
    var_29 = '2023-02-30T15:30:45'
    var_30 = var_0.validate(var_29)
    var_31 = '2023-13-01T15:30:45'
    var_32 = var_0.validate(var_31)
    var_33 = '2023-10-27T25:00:00'
    var_34 = var_0.validate(var_33)



# Parsed testcases at query #8
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:00'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 0
    var_5 = '08:30:45'
    var_6 = var_0.validate(var_5)
    var_7 = 8
    var_8 = 30
    var_9 = 45



# Parsed testcases at query #9
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '1999-12-31'
    var_6 = var_0.validate(var_5)
    var_7 = 1999
    var_8 = 12
    var_9 = 31
    var_10 = '2000-02-29'
    var_11 = var_0.validate(var_10)
    var_12 = 2000
    var_13 = 2
    var_14 = 29
    var_15 = '01-01-2023'
    var_16 = var_0.validate(var_15)
    var_17 = '2023/01/01'
    var_18 = var_0.validate(var_17)
    var_19 = 'not-a-date'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-02-30'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-13-01'
    var_24 = var_0.validate(var_23)
    var_25 = '2023-01-32'
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #10
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'https://www.google.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'https://www.google.com'
    var_3 = 'http://localhost:8080'
    var_4 = var_0.validate(var_3)
    assert var_4 == 'http://localhost:8080'
    var_5 = 'ftp://files.example.com'
    var_6 = var_0.validate(var_5)
    assert var_6 == 'ftp://files.example.com'
    var_7 = 'www.google.com'
    var_8 = var_0.validate(var_7)



# Parsed testcases at query #11
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:00'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 0
    var_5 = '09:05:30'
    var_6 = var_0.validate(var_5)
    var_7 = 9
    var_8 = 5
    var_9 = 30



# Parsed testcases at query #12
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name+tag@domain.co.uk'
    var_3 = '123456@example.org'
    var_4 = '"quoted-local-part"@example.com'
    var_5 = 'email@subdomain.example.com'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'plainaddress'
    var_8 = '#@%^%#$@#$@#.com'
    var_9 = '@example.com'
    var_10 = 'Joe Smith <email@example.com>'
    var_11 = 'email.example.com'
    var_12 = 'email@example@example.com'
    var_13 = '.email@example.com'
    var_14 = 'email.@example.com'
    var_15 = 'email..email@example.com'
    var_16 = 'email@example..com'
    var_17 = [var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16]
    var_18 = str(var_1)



# Parsed testcases at query #13
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:00'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 0
    var_5 = '23:59:59'
    var_6 = var_0.validate(var_5)
    var_7 = 23
    var_8 = 59
    var_9 = '08:30:15.123456'
    var_10 = var_0.validate(var_9)
    var_11 = 8
    var_12 = 30
    var_13 = 15
    var_14 = 123456
    var_15 = '08:30:15.123'
    var_16 = var_0.validate(var_15)
    var_17 = 123000
    var_18 = '0:0:0'
    var_19 = var_0.validate(var_18)
    var_20 = '12-00'
    var_21 = var_0.validate(var_20)
    var_22 = 'abc'
    var_23 = var_0.validate(var_22)
    var_24 = '25:00'
    var_25 = var_0.validate(var_24)
    var_26 = '12:61'
    var_27 = var_0.validate(var_26)
    var_28 = '12:00:61'
    var_29 = var_0.validate(var_28)



# Parsed testcases at query #14
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name+tag@domain.co.uk'
    var_3 = '1234567890@example.com'
    var_4 = 'email@subdomain.example.com'
    var_5 = '_______@example.com'
    var_6 = 'email@example-one.com'
    var_7 = '"very.unusual.@.unusual.com"@example.com'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = 'plainaddress'
    var_10 = '#@%^%#$@#$@#.com'
    var_11 = '@example.com'
    var_12 = 'Joe Smith <email@example.com>'
    var_13 = 'email.example.com'
    var_14 = 'email@example@example.com'
    var_15 = '.email@example.com'
    var_16 = 'email.@example.com'
    var_17 = 'email..email@example.com'
    var_18 = 'email@example..com'
    var_19 = 'Abc..123@example.com'
    var_20 = [var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19]



# Parsed testcases at query #15
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name@domain.co.uk'
    var_3 = 'first.last@sub.domain.org'
    var_4 = '123456@example.com'
    var_5 = 'email@example-one.com'
    var_6 = '_______@example.com'
    var_7 = 'email@example.name'
    var_8 = 'email@example.museum'
    var_9 = 'email@example.co.jp'
    var_10 = '"very.unusual.@.unusual.com"@example.com'
    var_11 = '"quoted"@example.com'
    var_12 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = 'plainaddress'
    var_14 = '#@%^%#$@#$@#.com'
    var_15 = '@example.com'
    var_16 = 'Joe Smith <email@example.com>'
    var_17 = 'email.example.com'
    var_18 = 'email@example@example.com'
    var_19 = '.email@example.com'
    var_20 = 'email.@example.com'
    var_21 = 'email..email@example.com'
    var_22 = 'あいうえお@example.com'
    var_23 = 'email@example.com (Joe Smith)'
    var_24 = 'email@example'
    var_25 = 'email@-example.com'
    var_26 = 'email@example..com'
    var_27 = [var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_24, var_25, var_26]
    var_28 = str(var_1)



# Parsed testcases at query #16
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.ip_address(var_1)
    var_4 = '127.0.0.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.ip_address(var_4)
    var_7 = '255.255.255.255'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.ip_address(var_7)
    var_10 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.ip_address(var_10)
    var_13 = 'not-an-ip'
    var_14 = var_0.validate(var_13)
    var_15 = '192.168.1'
    var_16 = var_0.validate(var_15)
    var_17 = ''
    var_18 = var_0.validate(var_17)



# Parsed testcases at query #17
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name+tag@domain.co.uk'
    var_3 = '123456@example.org'
    var_4 = 'email@subdomain.example.com'
    var_5 = '"quoted-string"@example.com'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'plainaddress'
    var_8 = '#@%^%#$@#$@#.com'
    var_9 = '@example.com'
    var_10 = 'Joe Smith <email@example.com>'
    var_11 = 'email.example.com'
    var_12 = 'email@example@example.com'
    var_13 = '.email@example.com'
    var_14 = 'email.@example.com'
    var_15 = 'email..email@example.com'
    var_16 = [var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15]
    var_17 = str(var_1)



# Parsed testcases at query #18
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T10:30:00Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 30
    var_7 = '2023-10-27 10:30:00+02:00'
    var_8 = var_0.validate(var_7)
    var_9 = 2
    var_10 = module_1.timedelta()
    var_11 = '2023-10-27T10:30:00-05:00'
    var_12 = var_0.validate(var_11)
    var_13 = -5
    var_14 = module_1.timedelta()
    var_15 = '2023-10-27T10:30:00.123456'
    var_16 = var_0.validate(var_15)
    var_17 = '2023-10-27 10:30:00'
    var_18 = var_0.validate(var_17)
    var_19 = '2023-10-27'
    var_20 = var_0.validate(var_19)
    var_21 = 'not-a-date'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-13-01T10:00:00'
    var_24 = var_0.validate(var_23)
    var_25 = '2023-10-32T10:00:00'
    var_26 = var_0.validate(var_25)
    var_27 = '2023-10-27T25:00:00'
    var_28 = var_0.validate(var_27)



# Parsed testcases at query #19
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = 2023
    var_2 = 10
    var_3 = 25
    var_4 = None
    var_5 = var_0.serialize(var_4)
    assert var_5 is None
    var_6 = 1999
    var_7 = 1
    var_8 = '2023-10-25'
    var_9 = var_0.serialize(var_8)



# Parsed testcases at query #20
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name+tag@domain.co.uk'
    var_3 = '123456@example.org'
    var_4 = '"quoted-local-part"@example.com'
    var_5 = 'simple@subdomain.example.com'
    var_6 = 'email@example.museum'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = 'plainaddress'
    var_9 = '#@%^%#$@#$@#.com'
    var_10 = '@example.com'
    var_11 = 'Joe Smith <email@example.com>'
    var_12 = 'email.example.com'
    var_13 = 'email@example@example.com'
    var_14 = '.email@example.com'
    var_15 = 'email.@example.com'
    var_16 = 'email..email@example.com'
    var_17 = 'email@example..com'
    var_18 = [var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17]



# Parsed testcases at query #21
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '1999-12-31'
    var_6 = var_0.validate(var_5)
    var_7 = 1999
    var_8 = 12
    var_9 = 31
    var_10 = '2000-02-29'
    var_11 = var_0.validate(var_10)
    var_12 = 2000
    var_13 = 2
    var_14 = 29
    var_15 = '01-01-2023'
    var_16 = var_0.validate(var_15)
    var_17 = '2023/01/01'
    var_18 = var_0.validate(var_17)
    var_19 = '2023-1-1'
    var_20 = var_0.validate(var_19)
    var_21 = 'not-a-date'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-02-30'
    var_24 = var_0.validate(var_23)
    var_25 = '2023-13-01'
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #22
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name@domain.co.uk'
    var_3 = 'first.last@sub.domain.org'
    var_4 = 'abc@example.museum'
    var_5 = '123@example.com'
    var_6 = 'email@domain-one.com'
    var_7 = '"quoted-string"@example.com'
    var_8 = 'simple@example.com'
    var_9 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8]
    var_10 = 'plainaddress'
    var_11 = '#@%^%#$@#$@#.com'
    var_12 = '@example.com'
    var_13 = 'Joe Smith <email@example.com>'
    var_14 = 'email.example.com'
    var_15 = 'email@example@example.com'
    var_16 = '.email@example.com'
    var_17 = 'email.@example.com'
    var_18 = 'email..email@example.com'
    var_19 = 'email@example..com'
    var_20 = 'Abc..123@example.com'
    var_21 = [var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20]
    var_22 = str(var_1)
    assert var_22 == 'Must be a valid email format.'



# Parsed testcases at query #23
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name+tag@domain.co.uk'
    var_3 = '1234567890@example.com'
    var_4 = 'email@subdomain.example.com'
    var_5 = '"quoted-string"@example.com'
    var_6 = 'very.common@example.com'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = 'plainaddress'
    var_9 = '#@%^%#$@#$@#.com'
    var_10 = '@example.com'
    var_11 = 'Joe Smith <email@example.com>'
    var_12 = 'email.example.com'
    var_13 = 'email@example@example.com'
    var_14 = '.email@example.com'
    var_15 = 'email.@example.com'
    var_16 = 'email..email@example.com'
    var_17 = 'email@example..com'
    var_18 = 'Abc..123@example.com'
    var_19 = [var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = str(var_1)



# Parsed testcases at query #24
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name+tag@domain.co.uk'
    var_3 = '1234567890@example.com'
    var_4 = 'email@subdomain.example.com'
    var_5 = '"quoted-string"@example.com'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = 'plainaddress'
    var_8 = '#@%^%#$@#$@#.com'
    var_9 = '@example.com'
    var_10 = 'Joe Smith <email@example.com>'
    var_11 = 'email.example.com'
    var_12 = 'email@example@example.com'
    var_13 = '.email@example.com'
    var_14 = 'email.@example.com'
    var_15 = 'email..email@example.com'
    var_16 = 'email@example..com'
    var_17 = [var_7, var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16]



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-10-25'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 25
    var_6 = '2023-1-5'
    var_7 = var_0.validate(var_6)
    var_8 = 1
    var_9 = 5
    var_10 = '2023/10/25'
    var_11 = var_0.validate(var_10)
    var_12 = '2023-10'
    var_13 = var_0.validate(var_12)
    var_14 = '2023-02-30'
    var_15 = var_0.validate(var_14)
    var_16 = '2023-13-01'
    var_17 = var_0.validate(var_16)
    var_18 = None
    var_19 = var_0.validate(var_18)



# Parsed testcases at query #2
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'test@example.com'
    var_3 = 'user.name+tag@domain.co.uk'
    var_4 = var_0.validate(var_3)
    assert var_4 == 'user.name+tag@domain.co.uk'
    var_5 = '123@abc.com'
    var_6 = var_0.validate(var_5)
    assert var_6 == '123@abc.com'
    var_7 = '"quoted-string"@example.com'
    var_8 = var_0.validate(var_7)
    assert var_8 == '"quoted-string"@example.com'
    var_9 = 'plainaddress'
    var_10 = '#@%^%#%@#@#.com'
    var_11 = '@example.com'
    var_12 = 'Joe Smith <email@example.com>'
    var_13 = 'email.example.com'
    var_14 = 'email@example@example.com'
    var_15 = '.email@example.com'
    var_16 = 'email.@example.com'
    var_17 = 'email..email@example.com'
    var_18 = 'email@example..com'
    var_19 = [var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18]
    var_20 = str(var_1)
    assert var_20 == 'Must be a valid email format.'



# Parsed testcases at query #3
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '2024-02-29'
    var_6 = var_0.validate(var_5)
    var_7 = 2024
    var_8 = 2
    var_9 = 29



# Parsed testcases at query #4
#--------------------------


import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-567812345678'
    var_2 = module_1.UUID(var_1)
    var_3 = var_0.serialize(var_2)
    assert var_3 == '12345678-1234-5678-1234-567812345678'
    var_4 = None
    var_5 = var_0.serialize(var_4)
    assert var_5 is None
    var_6 = module_1.uuid4()
    var_7 = str(var_6)
    var_8 = 'not-a-uuid-object'
    var_9 = var_0.serialize(var_8)



# Parsed testcases at query #5
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'https://google.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'https://google.com'
    var_3 = 'http://localhost:8080'
    var_4 = var_0.validate(var_3)
    assert var_4 == 'http://localhost:8080'
    var_5 = 'ftp://files.example.com/path'
    var_6 = var_0.validate(var_5)
    assert var_6 == 'ftp://files.example.com/path'
    var_7 = 'google.com'
    var_8 = var_0.validate(var_7)
    var_9 = 'https://'
    var_10 = var_0.validate(var_9)
    var_11 = ''
    var_12 = var_0.validate(var_11)



# Parsed testcases at query #6
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = var_0.validate(var_1)
    var_3 = '19ass.168.1.1'
    var_4 = module_1.IPv4Address(var_3)
    var_5 = '127.0.0.1'
    var_6 = var_0.validate(var_5)
    var_7 = module_1.IPv4Address(var_5)
    var_8 = '0.0.0.0'
    var_9 = var_0.validate(var_8)
    var_10 = module_1.IPv4Address(var_8)
    var_11 = '255.255.255.255'
    var_12 = var_0.validate(var_11)
    var_13 = module_1.IPv4Address(var_11)
    var_14 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_15 = var_0.validate(var_14)
    var_16 = module_1.IPv6Address(var_14)
    var_17 = 'not-an-ip'
    var_18 = var_0.validate(var_17)
    var_19 = '192.168.1'
    var_20 = var_0.validate(var_19)



# Parsed testcases at query #7
#--------------------------


import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '550e8400-e29b-41d4-a716-446655440000'
    var_2 = '6ba7b810-9dad-11d1-80b4-00c04fd430c8'
    var_3 = var_0.validate(var_2)
    var_4 = module_1.UUID(var_2)
    var_5 = 'not-a-uuid'
    var_6 = var_0.validate(var_5)
    var_7 = '550e8400-e29b-41d4-a716'
    var_8 = var_0.validate(var_7)
    var_9 = '550e8400-e29b-61d4-a716-446655440000'
    var_10 = var_0.validate(var_9)



# Parsed testcases at query #8
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = var_0.validate(var_1)
    var_3 = '19jack.168.1.1'
    var_4 = 'jack'
    var_5 = ''
    var_6 = '127.0.0.1'
    var_7 = var_0.validate(var_6)
    var_8 = module_1.IPv4Address(var_6)
    var_9 = '255.255.255.255'
    var_10 = var_0.validate(var_9)
    var_11 = module_1.IPv4Address(var_9)
    var_12 = '0.0.0.0'
    var_13 = var_0.validate(var_12)
    var_14 = module_1.IPv4Address(var_12)
    var_15 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_16 = var_0.validate(var_15)
    var_17 = module_1.IPv6Address(var_15)
    var_18 = '1:2:3:4:5:6:7:8'
    var_19 = var_0.validate(var_18)
    var_20 = module_1.IPv6Address(var_18)
    var_21 = 'not-an-ip'
    var_22 = var_0.validate(var_21)
    var_23 = '192.168.1'
    var_24 = var_0.validate(var_23)
    var_25 = '999.999.999.999'
    var_26 = var_0.validate(var_25)
    var_27 = '::1'
    var_28 = var_0.validate(var_27)



# Parsed testcases at query #9
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '0.0.0.0'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '255.255.255.255'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = '2001:db8:85a3::8a2e:370:7334'
    var_16 = module_1.IPv6Address(var_15)
    var_17 = '1:2:3:4:5:6:7:8'
    var_18 = var_0.validate(var_17)
    var_19 = module_1.IPv6Address(var_17)
    var_20 = 'not-an-ip'
    var_21 = var_0.validate(var_20)
    var_22 = '127.0.0.256'
    var_23 = var_0.validate(var_22)
    var_24 = 'abc.def.ghi.jkl'
    var_25 = var_0.validate(var_24)



# Parsed testcases at query #10
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30
    var_5 = '09:05:01'
    var_6 = var_0.validate(var_5)
    var_7 = 9
    var_8 = 5
    var_9 = 1
    var_10 = '23:59:59.123456'
    var_11 = var_0.validate(var_10)
    var_12 = 23
    var_13 = 59
    var_14 = 123456
    var_15 = '00:00:00.1'
    var_16 = var_0.validate(var_15)
    var_17 = 0
    var_18 = 100000
    var_19 = '12-30'
    var_20 = var_0.validate(var_19)
    var_21 = 'abc'
    var_22 = var_0.validate(var_21)
    var_23 = '25:00:00'
    var_24 = var_0.validate(var_23)
    var_25 = '12:61:00'
    var_26 = var_0.validate(var_25)
    var_27 = '12:00:61'
    var_28 = var_0.validate(var_27)



# Parsed testcases at query #11
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:00Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = '2023-10-27 15:30:00'
    var_9 = var_0.validate(var_8)
    var_10 = '2023-10-27T15:30:00.123456'
    var_11 = var_0.validate(var_10)
    var_12 = '2023-10-27T15:30:00+02:00'
    var_13 = var_0.validate(var_12)
    var_14 = 2
    var_15 = module_1.timedelta()
    var_16 = '2023-10-27T15:30:00-05:00'
    var_17 = var_0.validate(var_16)
    var_18 = -5
    var_19 = module_1.timedelta()
    var_20 = '27-10-2023 15:30:00'
    var_21 = var_0.validate(var_20)
    var_22 = '2023-13-01T15:30:00'
    var_23 = var_0.validate(var_22)
    var_24 = '2023-02-30T15:30:00'
    var_25 = var_0.validate(var_24)
    var_26 = '2023-10-27T15:30:00.1234567'
    var_27 = var_0.validate(var_26)



# Parsed testcases at query #12
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = '2023-10-27 15:30:45'
    var_10 = var_0.validate(var_9)
    var_11 = None
    var_12 = '2023-10-27T15:30:45+02:00'
    var_13 = var_0.validate(var_12)
    var_14 = 2
    var_15 = module_1.timedelta()
    var_16 = '2023-10-27T15:30:45-05:00'
    var_17 = var_0.validate(var_16)
    var_18 = -5
    var_19 = module_1.timedelta()
    var_20 = '2023-10-27T15:30:45.123456'
    var_21 = var_0.validate(var_20)
    var_22 = '27-10-2023 15:30:45'
    var_23 = var_0.validate(var_22)
    var_24 = '2023-02-30T15:30:45'
    var_25 = var_0.validate(var_24)
    var_26 = '2023-10-27T15:30:45+99:00'
    var_27 = var_0.validate(var_26)



# Parsed testcases at query #13
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '127.0.0.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.ip_address(var_1)
    var_4 = '192.168.1.1'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.ip_address(var_4)
    var_7 = '0.0.0.0'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.ip_address(var_7)
    var_10 = '255.255.255.255'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.ip_address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.ip_address(var_13)
    var_16 = '1:2:3:4:5:6:7:8'
    var_17 = var_0.validate(var_16)
    var_18 = module_1.ip_address(var_16)
    var_19 = 'not-an-ip'
    var_20 = var_0.validate(var_19)
    var_21 = '127.0.0.256'
    var_22 = var_0.validate(var_21)
    var_23 = '127.0.0.256'
    var_24 = var_0.validate(var_23)
    var_25 = '127.0.0.1.1'
    var_26 = var_0.validate(var_25)
    var_27 = 'abc.def.ghi.jkl'
    var_28 = var_0.validate(var_27)



# Parsed testcases at query #14
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:00Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = '2023-10-27 15:30:00'
    var_9 = var_0.validate(var_8)
    var_10 = '2023-10-27T15:30:00+02:00'
    var_11 = var_0.validate(var_10)
    var_12 = 2
    var_13 = module_1.timedelta()
    var_14 = '2023-10-27T15:30:00-05:00'
    var_15 = var_0.validate(var_14)
    var_16 = -5
    var_17 = module_1.timedelta()
    var_18 = '2023-10-27T15:30:00.123456'
    var_19 = var_0.validate(var_18)
    var_20 = '2023-10-27T15:30:00.12'
    var_21 = var_0.validate(var_20)
    var_22 = '27-10-2023 15:30:00'
    var_23 = var_0.validate(var_22)
    var_24 = 'not-a-date'
    var_25 = var_0.validate(var_24)
    var_26 = '2023-02-30T15:30:00'
    var_27 = var_0.validate(var_26)
    var_28 = '2023-13-01T15:30:00'
    var_29 = var_0.validate(var_28)
    var_30 = '2023-10-27T25:00:00'
    var_31 = var_0.validate(var_30)



# Parsed testcases at query #15
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-10-25'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 25
    var_6 = '2023-1-5'
    var_7 = var_0.validate(var_6)
    var_8 = 1
    var_9 = 5
    var_10 = '2023/10/25'
    var_11 = var_0.validate(var_10)
    var_12 = 'abcd-ef-gh'
    var_13 = var_0.validate(var_12)
    var_14 = '2023-02-30'
    var_15 = var_0.validate(var_14)
    var_16 = '2023-13-01'
    var_17 = var_0.validate(var_16)
    var_18 = '2021-02-29'
    var_19 = var_0.validate(var_18)
    var_20 = '2024-02-29'
    var_21 = var_0.validate(var_20)
    var_22 = 2024
    var_23 = 2
    var_24 = 29



# Parsed testcases at query #16
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = '2023-10-27 15:30:45'
    var_10 = var_0.validate(var_9)
    var_11 = None
    var_12 = '2023-10-27T15:30:45.123456'
    var_13 = var_0.validate(var_12)
    var_14 = '2023-10-27T15:30:45+02:00'
    var_15 = var_0.validate(var_14)
    var_16 = 2
    var_17 = module_1.timedelta()
    var_18 = '2023-10-27T15:30:45-05:00'
    var_19 = var_0.validate(var_18)
    var_20 = -5
    var_21 = module_1.timedelta()
    var_22 = '27-10-2023 15:30:45'
    var_23 = var_0.validate(var_22)
    var_24 = '2023-13-01T15:30:45'
    var_25 = var_0.validate(var_24)
    var_26 = '2023-02-30T15:30:45'
    var_27 = var_0.validate(var_26)



# Parsed testcases at query #17
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-10-27'
    var_2 = 2023
    var_3 = 10
    var_4 = 27
    var_5 = var_0.validate(var_1)
    var_6 = '2023-1-5'
    var_7 = 1
    var_8 = 5
    var_9 = var_0.validate(var_6)
    var_10 = '2023/10/27'
    var_11 = var_0.validate(var_10)
    var_12 = '2023-10-27 12:00:00'
    var_13 = var_0.validate(var_12)
    var_14 = '2023-02-30'
    var_15 = var_0.validate(var_14)
    var_16 = '2023-13-01'
    var_17 = var_0.validate(var_16)
    var_18 = None
    var_19 = var_0.validate(var_18)



# Parsed testcases at query #18
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-01-01'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 1
    var_5 = '1999-12-31'
    var_6 = var_0.validate(var_5)
    var_7 = 1999
    var_8 = 12
    var_9 = 31
    var_10 = '2000-02-29'
    var_11 = var_0.validate(var_10)
    var_12 = 2000
    var_13 = 2
    var_14 = 29
    var_15 = '01-01-2023'
    var_16 = var_0.validate(var_15)
    var_17 = '202le-01-01'
    var_18 = var_0.validate(var_17)
    var_19 = '2023/01/01'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-13-01'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-02-29'
    var_24 = var_0.validate(var_23)
    var_25 = '2023-04-31'
    var_26 = var_0.validate(var_25)
    var_27 = '0000-00-00'
    var_28 = var_0.validate(var_27)



# Parsed testcases at query #19
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-27T15:30:45Z'
    var_2 = var_0.validate(var_1)
    var_3 = 2023
    var_4 = 10
    var_5 = 27
    var_6 = 15
    var_7 = 30
    var_8 = 45
    var_9 = '2023-10-27 15:30:45+02:00'
    var_10 = var_0.validate(var_9)
    var_11 = 2
    var_12 = module_1.timedelta()
    var_13 = '2023-10-27 15:30:45-05:00'
    var_14 = var_0.validate(var_13)
    var_15 = -5
    var_16 = module_1.timedelta()
    var_17 = '2023-10-27T15:30:45.123456Z'
    var_18 = var_0.validate(var_17)
    var_19 = '2023-10-27 15:30'
    var_20 = var_0.validate(var_19)
    var_21 = '27-10-2023 15:30'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-02-30T15:30:00Z'
    var_24 = var_0.validate(var_23)
    var_25 = '2023-10-27 25:00:00'
    var_26 = var_0.validate(var_25)



