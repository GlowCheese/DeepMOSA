####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.formats as module_0
import uuid as module_1
import ipaddress as module_2

def test_case_0():
    var_0 = module_0.BaseFormat()
    var_1 = 'test'
    var_2 = var_0.is_native_type(var_1)
    var_3 = module_0.DateFormat()
    var_4 = 2023
    var_5 = 1
    var_6 = '2023-01-01'
    var_7 = var_3.is_native_type(var_6)
    assert var_7 is False
    var_8 = None
    var_9 = var_3.is_native_type(var_8)
    assert var_9 is False
    var_10 = 123
    var_11 = var_3.is_native_type(var_10)
    assert var_11 is False
    var_12 = module_0.TimeFormat()
    var_13 = 12
    var_14 = 30
    var_15 = 45
    var_16 = '12:30:45'
    var_17 = var_12.is_native_type(var_16)
    assert var_17 is False
    var_18 = module_0.DateTimeFormat()
    var_19 = '2023-01-01T12:30:45'
    var_20 = var_18.is_native_type(var_19)
    assert var_20 is False
    var_21 = module_0.UUIDFormat()
    var_22 = module_1.uuid4()
    var_23 = var_21.is_native_type(var_22)
    assert var_23 is True
    var_24 = str(var_22)
    var_25 = var_21.is_native_type(var_24)
    assert var_25 is False
    var_26 = var_21.is_native_type(var_10)
    assert var_26 is False
    var_27 = module_0.EmailFormat()
    var_28 = 'test@example.com'
    var_29 = var_27.is_native_type(var_28)
    assert var_29 is False
    var_30 = var_27.is_native_type(var_8)
    assert var_30 is False
    var_31 = var_27.is_native_type(var_10)
    assert var_31 is False
    var_32 = module_0.IPAddressFormat()
    var_33 = '192.168.1.1'
    var_34 = module_2.IPv4Address(var_33)
    var_35 = var_32.is_native_type(var_34)
    assert var_35 is True
    var_36 = '::1'
    var_37 = module_2.IPv6Address(var_36)
    var_38 = var_32.is_native_type(var_37)
    assert var_38 is True
    var_39 = var_32.is_native_type(var_33)
    assert var_39 is False
    var_40 = var_32.is_native_type(var_8)
    assert var_40 is False
    var_41 = module_0.URLFormat()
    var_42 = 'https://example.com'
    var_43 = var_41.is_native_type(var_42)
    assert var_43 is False
    var_44 = var_41.is_native_type(var_8)
    assert var_44 is False
    var_45 = var_41.is_native_type(var_10)
    assert var_45 is False



# Parsed testcases at query #2
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30
    var_5 = '09:45'
    var_6 = var_0.validate(var_5)
    var_7 = 9
    var_8 = 45
    var_9 = '23:59'
    var_10 = var_0.validate(var_9)
    var_11 = 23
    var_12 = 59
    var_13 = '00:00'
    var_14 = var_0.validate(var_13)
    var_15 = 0
    var_16 = '12:30:45'
    var_17 = var_0.validate(var_16)
    var_18 = '09:45:30'
    var_19 = var_0.validate(var_18)
    var_20 = '12:30:45.123456'
    var_21 = var_0.validate(var_20)
    var_22 = 123456
    var_23 = '09:45:30.123'
    var_24 = var_0.validate(var_23)
    var_25 = 123000
    var_26 = '12:30:45.1'
    var_27 = var_0.validate(var_26)
    var_28 = 100000
    var_29 = '25:30'
    var_30 = var_0.validate(var_29)
    var_31 = '12:60'
    var_32 = var_0.validate(var_31)
    var_33 = '12:30:60'
    var_34 = var_0.validate(var_33)
    var_35 = '12:30:45.1234567'
    var_36 = var_0.validate(var_35)
    var_37 = 'not a time'
    var_38 = var_0.validate(var_37)
    var_39 = '12'
    var_40 = var_0.validate(var_39)
    var_41 = '12:30:45:67'
    var_42 = var_0.validate(var_41)
    var_43 = '24:00'
    var_44 = var_0.validate(var_43)
    var_45 = '12:60:45'
    var_46 = var_0.validate(var_45)
    var_47 = '12:30:60'
    var_48 = var_0.validate(var_47)
    var_49 = '12:30:45.9999999'
    var_50 = var_0.validate(var_49)



# Parsed testcases at query #3
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
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = module_1.timedelta()
    var_11 = -2
    var_12 = module_1.timedelta()
    var_13 = module_1.timedelta()
    var_14 = 0
    var_15 = 123
    var_16 = 23
    var_17 = 59
    var_18 = 999999



# Parsed testcases at query #4
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = '127.0.0.1'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_10 = '2001:db8:85a3::8a2e:370:7334'
    var_11 = 'fe80::1'
    var_12 = '::1'
    var_13 = '2001:0db8:0000:0000:0000:0000:0000:0001'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = str(var_3)
    var_16 = 'not an ip'
    var_17 = '256.256.256.256'
    var_18 = '192.168.1'
    var_19 = '192.168.1.1.1'
    var_20 = '192.168.1.256'
    var_21 = '2001:db8:85a3::8a2e:370:7334:extra'
    var_22 = '2001::db8::1'
    var_23 = [var_16, var_17, var_18, var_19, var_20, var_21, var_22]
    var_24 = '999.999.999.999'
    var_25 = '300.300.300.300'
    var_26 = [var_24, var_25]
    var_27 = module_1.IPv4Address(var_1)
    var_28 = var_0.is_native_type(var_27)
    assert var_28 is True
    var_29 = module_1.IPv6Address(var_12)
    var_30 = var_0.is_native_type(var_29)
    assert var_30 is True
    var_31 = var_0.is_native_type(var_1)
    assert var_31 is False
    var_32 = 123
    var_33 = var_0.is_native_type(var_32)
    assert var_33 is False
    var_34 = None
    var_35 = var_0.is_native_type(var_34)
    assert var_35 is False
    var_36 = var_0.serialize(var_34)
    assert var_36 is None
    var_37 = module_1.IPv4Address(var_1)
    var_38 = var_0.serialize(var_37)
    assert var_38 == '192.168.1.1'
    var_39 = module_1.IPv6Address(var_12)
    var_40 = var_0.serialize(var_39)
    assert var_40 == '::1'
    var_41 = 'not an ip'
    var_42 = var_0.serialize(var_41)



# Parsed testcases at query #5
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-05T14:30:00'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-10-05 14:30:00'
    var_4 = var_0.validate(var_3)
    var_5 = '2023-10-05T14:30:00.123456'
    var_6 = var_0.validate(var_5)
    var_7 = '2023-10-05T14:30:00.123'
    var_8 = var_0.validate(var_7)
    var_9 = '2023-10-05T14:30:00Z'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-10-05T14:30:00+05:30'
    var_12 = var_0.validate(var_11)
    var_13 = None
    var_14 = 5
    var_15 = 30
    var_16 = module_1.timedelta()
    var_17 = '2023-10-05T14:30:00-03:00'
    var_18 = var_0.validate(var_17)
    var_19 = -3
    var_20 = module_1.timedelta()
    var_21 = '2023-10-05T14:30:00+0530'
    var_22 = var_0.validate(var_21)
    var_23 = module_1.timedelta()
    var_24 = '2023-10-05T14:30:00+05'
    var_25 = var_0.validate(var_24)
    var_26 = module_1.timedelta()
    var_27 = '2023-10-05'
    var_28 = var_0.validate(var_27)
    var_29 = '10/05/2023 14:30:00'
    var_30 = var_0.validate(var_29)
    var_31 = '2023-13-05T14:30:00'
    var_32 = var_0.validate(var_31)
    var_33 = '2023-10-05T25:30:00'
    var_34 = var_0.validate(var_33)
    var_35 = '2023-02-30T14:30:00'
    var_36 = var_0.validate(var_35)
    var_37 = 2023
    var_38 = 10
    var_39 = 14
    var_40 = 0



# Parsed testcases at query #6
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-1234-1234-123456789abc'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = '12345678-1234-6234-1234-123456789abc'
    var_5 = var_0.validate(var_4)
    var_6 = 'not-a-uuid'
    var_7 = var_0.validate(var_6)
    var_8 = '12345678-1234-1234-1234-123456789abg'
    var_9 = var_0.validate(var_8)
    var_10 = '12345678-1234-1234-1234-123456789ABC'
    var_11 = var_0.validate(var_10)
    var_12 = str(var_11)
    var_13 = '12345678-1234-1234-8234-123456789abc'
    var_14 = var_0.validate(var_13)
    var_15 = str(var_14)
    var_16 = '12345678-1234-3234-8234-123456789abc'
    var_17 = var_0.validate(var_16)
    var_18 = str(var_17)
    var_19 = '12345678-1234-4234-8234-123456789abc'
    var_20 = var_0.validate(var_19)
    var_21 = str(var_20)
    var_22 = '12345678-1234-5234-8234-123456789abc'
    var_23 = var_0.validate(var_22)
    var_24 = str(var_23)



# Parsed testcases at query #7
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30
    var_5 = '09:45'
    var_6 = var_0.validate(var_5)
    var_7 = 9
    var_8 = 45
    var_9 = '23:59'
    var_10 = var_0.validate(var_9)
    var_11 = 23
    var_12 = 59
    var_13 = '00:00'
    var_14 = var_0.validate(var_13)
    var_15 = 0
    var_16 = '12:30:45'
    var_17 = var_0.validate(var_16)
    var_18 = '09:45:30'
    var_19 = var_0.validate(var_18)
    var_20 = '12:30:45.123'
    var_21 = var_0.validate(var_20)
    var_22 = 123000
    var_23 = '12:30:45.123456'
    var_24 = var_0.validate(var_23)
    var_25 = 123456
    var_26 = '12:30:45.1'
    var_27 = var_0.validate(var_26)
    var_28 = 100000
    var_29 = '12:30:45.12'
    var_30 = var_0.validate(var_29)
    var_31 = 120000
    var_32 = '12:30:45.12345'
    var_33 = var_0.validate(var_32)
    var_34 = 123450
    var_35 = '25:00'
    var_36 = var_0.validate(var_35)
    var_37 = '12:60'
    var_38 = var_0.validate(var_37)
    var_39 = '12:30:60'
    var_40 = var_0.validate(var_39)
    var_41 = '12:30:45.1234567'
    var_42 = var_0.validate(var_41)
    var_43 = 'invalid'
    var_44 = var_0.validate(var_43)
    var_45 = '12:30:45.'
    var_46 = var_0.validate(var_45)
    var_47 = '12:30:'
    var_48 = var_0.validate(var_47)
    var_49 = '12:'
    var_50 = var_0.validate(var_49)
    var_51 = '24:00'
    var_52 = var_0.validate(var_51)
    var_53 = '12:60'
    var_54 = var_0.validate(var_53)
    var_55 = '12:30:60'
    var_56 = var_0.validate(var_55)
    var_57 = '1:2'
    var_58 = var_0.validate(var_57)
    var_59 = 1
    var_60 = 2
    var_61 = '1:2:3'
    var_62 = var_0.validate(var_61)
    var_63 = 3
    var_64 = '1:2:3.4'
    var_65 = var_0.validate(var_64)
    var_66 = 400000



# Parsed testcases at query #8
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-05-15T14:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-05-15 14:30:45'
    var_4 = var_0.validate(var_3)
    var_5 = '2023-05-15T14:30:45.123456'
    var_6 = var_0.validate(var_5)
    var_7 = '2023-05-15T14:30:45.123'
    var_8 = var_0.validate(var_7)
    var_9 = '2023-05-15T14:30:45Z'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-05-15T14:30:45+05:30'
    var_12 = var_0.validate(var_11)
    var_13 = None
    var_14 = 5
    var_15 = 30
    var_16 = module_1.timedelta()
    var_17 = '2023-05-15T14:30:45-03:00'
    var_18 = var_0.validate(var_17)
    var_19 = -3
    var_20 = module_1.timedelta()
    var_21 = '2023-05-15T14:30:45+0530'
    var_22 = var_0.validate(var_21)
    var_23 = module_1.timedelta()
    var_24 = '2023-05-15T14:30:45+05'
    var_25 = var_0.validate(var_24)
    var_26 = module_1.timedelta()
    var_27 = '2023-05-15'
    var_28 = var_0.validate(var_27)
    var_29 = '2023-05-15 14-30-45'
    var_30 = var_0.validate(var_29)
    var_31 = '2023-13-15T14:30:45'
    var_32 = var_0.validate(var_31)
    var_33 = '2023-05-32T14:30:45'
    var_34 = var_0.validate(var_33)
    var_35 = '2023-05-15T24:30:45'
    var_36 = var_0.validate(var_35)
    var_37 = '2023-02-30T14:30:45'
    var_38 = var_0.validate(var_37)
    var_39 = 2023
    var_40 = 15
    var_41 = 14
    var_42 = 45
    var_43 = var_0.is_native_type(var_37)
    assert var_43 is False
    var_44 = 123
    var_45 = var_0.is_native_type(var_44)
    assert var_45 is False



# Parsed testcases at query #9
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '8.8.8.8'
    var_3 = '255.255.255.255'
    var_4 = '0.0.0.0'
    var_5 = '127.0.0.1'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_8 = 'fe80::1'
    var_9 = '::1'
    var_10 = '2001:db8::1'
    var_11 = [var_7, var_8, var_9, var_10]
    var_12 = 'not an ip'
    var_13 = '192.168.1'
    var_14 = '192.168.1.256'
    var_15 = '192.168.1.1.1'
    var_16 = '2001:db8:85a3::8a2e:0370:7334:extra'
    var_17 = [var_12, var_13, var_14, var_15, var_16]
    var_18 = '999.999.999.999'
    var_19 = '256.256.256.256'
    var_20 = [var_18, var_19]
    var_21 = module_1.IPv4Address(var_1)
    var_22 = var_0.is_native_type(var_21)
    assert var_22 is True
    var_23 = module_1.IPv6Address(var_9)
    var_24 = var_0.is_native_type(var_23)
    assert var_24 is True
    var_25 = var_0.is_native_type(var_1)
    assert var_25 is False
    var_26 = 123
    var_27 = var_0.is_native_type(var_26)
    assert var_27 is False
    var_28 = None
    var_29 = var_0.serialize(var_28)
    assert var_29 is None
    var_30 = module_1.IPv4Address(var_1)
    var_31 = var_0.serialize(var_30)
    assert var_31 == '192.168.1.1'
    var_32 = module_1.IPv6Address(var_9)
    var_33 = var_0.serialize(var_32)
    assert var_33 == '::1'



# Parsed testcases at query #10
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2023
    var_4 = 12
    var_5 = 25
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = 5
    var_11 = module_1.timedelta()
    var_12 = -3
    var_13 = module_1.timedelta()
    var_14 = module_1.timedelta()
    var_15 = 0
    var_16 = 123



# Parsed testcases at query #11
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-2-5'
    var_4 = var_0.validate(var_3)
    var_5 = '2023/12/25'
    var_6 = var_0.validate(var_5)
    var_7 = '2023-12-25T10:30:00'
    var_8 = var_0.validate(var_7)
    var_9 = '2023-13-25'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-02-30'
    var_12 = var_0.validate(var_11)
    var_13 = '2023-02-29'
    var_14 = var_0.validate(var_13)
    var_15 = '2024-02-29'
    var_16 = var_0.validate(var_15)
    var_17 = '2023-01-01'
    var_18 = var_0.validate(var_17)
    var_19 = '0001-01-01'
    var_20 = var_0.validate(var_19)
    var_21 = '9999-12-31'
    var_22 = var_0.validate(var_21)
    var_23 = 12345
    var_24 = var_0.validate(var_23)
    var_25 = ''
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #12
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = '127.0.0.1'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_10 = 'fe80::1'
    var_11 = '::1'
    var_12 = '2001:db8::1'
    var_13 = 'ff02::1'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'not an ip'
    var_16 = '192.168.1'
    var_17 = '192.168.1.1.1'
    var_18 = '192.168.1.256'
    var_19 = '192.168.1.-1'
    var_20 = '2001:db8:85a3::8a2e:0370:7334:extra'
    var_21 = 'gggg::1'
    var_22 = [var_15, var_16, var_17, var_18, var_19, var_20, var_21]
    var_23 = '999.999.999.999'
    var_24 = '256.256.256.256'
    var_25 = [var_23, var_24]
    var_26 = module_1.IPv4Address(var_1)
    var_27 = var_0.is_native_type(var_26)
    assert var_27 is True
    var_28 = module_1.IPv6Address(var_11)
    var_29 = var_0.is_native_type(var_28)
    assert var_29 is True
    var_30 = var_0.is_native_type(var_1)
    assert var_30 is False
    var_31 = 123
    var_32 = var_0.is_native_type(var_31)
    assert var_32 is False
    var_33 = None
    var_34 = var_0.is_native_type(var_33)
    assert var_34 is False
    var_35 = module_1.IPv4Address(var_1)
    var_36 = module_1.IPv6Address(var_11)
    var_37 = var_0.serialize(var_35)
    assert var_37 == '192.168.1.1'
    var_38 = var_0.serialize(var_36)
    assert var_38 == '::1'
    var_39 = var_0.serialize(var_33)
    assert var_39 is None



# Parsed testcases at query #13
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
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = module_1.timedelta()
    var_11 = -5
    var_12 = module_1.timedelta()
    var_13 = 0
    var_14 = module_1.timedelta()



# Parsed testcases at query #14
#--------------------------


import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-1234-1234-123456789abc'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = 'a1b2c3d4-e5f6-1a2b-3c4d-5e6f7a8b9c0d'
    var_5 = var_0.validate(var_4)
    var_6 = str(var_5)
    var_7 = 'not-a-uuid'
    var_8 = var_0.validate(var_7)
    var_9 = '12345678-1234-1234-1234-123456789abg'
    var_10 = var_0.validate(var_9)
    var_11 = '12345678-1234-1234-1234-123456789abc-extra'
    var_12 = var_0.validate(var_11)
    var_13 = '12345678123412341234123456789abc'
    var_14 = var_0.validate(var_13)
    var_15 = '12345678-1234-6234-1234-123456789abc'
    var_16 = var_0.validate(var_15)
    var_17 = '12345678-1234-1234-6234-123456789abc'
    var_18 = var_0.validate(var_17)
    var_19 = module_1.uuid4()
    var_20 = str(var_19)
    var_21 = var_0.validate(var_20)



# Parsed testcases at query #15
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-1-5'
    var_4 = var_0.validate(var_3)
    var_5 = '2023-12'
    var_6 = var_0.validate(var_5)
    var_7 = '2023/12/25'
    var_8 = var_0.validate(var_7)
    var_9 = '2023-12-25T10:30:00'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-13-25'
    var_12 = var_0.validate(var_11)
    var_13 = '2023-12-32'
    var_14 = var_0.validate(var_13)
    var_15 = '2023-02-30'
    var_16 = var_0.validate(var_15)
    var_17 = '2024-02-29'
    var_18 = var_0.validate(var_17)
    var_19 = '2023-02-29'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-01-01'
    var_22 = var_0.validate(var_21)
    var_23 = 2023
    var_24 = 12
    var_25 = 25
    var_26 = var_0.is_native_type(var_19)
    assert var_26 is False
    var_27 = 123
    var_28 = var_0.is_native_type(var_27)
    assert var_28 is False
    var_29 = None
    var_30 = var_0.is_native_type(var_29)
    assert var_30 is False



# Parsed testcases at query #16
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name@domain.co.uk'
    var_3 = 'user+tag@example.org'
    var_4 = 'user@sub.domain.com'
    var_5 = 'a@b.cd'
    var_6 = 'user@example.io'
    var_7 = 'USER@EXAMPLE.COM'
    var_8 = 'user123@example.com'
    var_9 = 'first.last@company.name'
    var_10 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = 'notanemail'
    var_12 = '@example.com'
    var_13 = 'user@'
    var_14 = 'user@.com'
    var_15 = 'user@example.'
    var_16 = 'user@example..com'
    var_17 = 'user name@example.com'
    var_18 = 'user@-example.com'
    var_19 = 'user@example-.com'
    var_20 = ''
    var_21 = '   '
    var_22 = 'user@example.c'
    var_23 = [var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22]
    var_24 = var_0.is_native_type(var_1)
    assert var_24 is False
    var_25 = None
    var_26 = var_0.is_native_type(var_25)
    assert var_26 is False
    var_27 = 123
    var_28 = var_0.is_native_type(var_27)
    assert var_28 is False
    var_29 = var_0.serialize(var_1)
    assert var_29 == 'test@example.com'
    var_30 = var_0.serialize(var_25)
    assert var_30 is None
    var_31 = var_0.serialize(var_20)
    assert var_31 == ''



# Parsed testcases at query #17
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-1-5'
    var_4 = var_0.validate(var_3)
    var_5 = '2023-12'
    var_6 = var_0.validate(var_5)
    var_7 = '2023/12/25'
    var_8 = var_0.validate(var_7)
    var_9 = '2023-12-25T10:30:00'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-13-25'
    var_12 = var_0.validate(var_11)
    var_13 = '2023-12-32'
    var_14 = var_0.validate(var_13)
    var_15 = '2023-02-30'
    var_16 = var_0.validate(var_15)
    var_17 = '2024-02-29'
    var_18 = var_0.validate(var_17)
    var_19 = '2023-02-29'
    var_20 = var_0.validate(var_19)
    var_21 = '0001-01-01'
    var_22 = var_0.validate(var_21)
    var_23 = '9999-12-31'
    var_24 = var_0.validate(var_23)
    var_25 = '2023-01-01'
    var_26 = var_0.validate(var_25)
    var_27 = '2023-12-25 '
    var_28 = var_0.validate(var_27)
    var_29 = ' 2023-12-25'
    var_30 = var_0.validate(var_29)



# Parsed testcases at query #18
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-2-5'
    var_4 = var_0.validate(var_3)
    var_5 = '2023/12/25'
    var_6 = var_0.validate(var_5)
    var_7 = '2023-12-25T10:30:00'
    var_8 = var_0.validate(var_7)
    var_9 = '2023-02-30'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-04-31'
    var_12 = var_0.validate(var_11)
    var_13 = '2023-13-01'
    var_14 = var_0.validate(var_13)
    var_15 = '2023-12-00'
    var_16 = var_0.validate(var_15)
    var_17 = '2024-02-29'
    var_18 = var_0.validate(var_17)
    var_19 = '2023-02-29'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-01-09'
    var_22 = var_0.validate(var_21)
    var_23 = '0001-01-01'
    var_24 = var_0.validate(var_23)
    var_25 = '9999-12-31'
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #19
#--------------------------


import typesystem.formats as module_0
import uuid as module_1

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = '12345678-1234-5678-1234-567812345678'
    var_4 = module_1.UUID(var_3)
    var_5 = var_0.serialize(var_4)
    assert var_5 == '12345678-1234-5678-1234-567812345678'
    var_6 = '00000000-0000-0000-0000-000000000000'
    var_7 = module_1.UUID(var_6)
    var_8 = var_0.serialize(var_7)
    assert var_8 == '00000000-0000-0000-0000-000000000000'
    var_9 = 'ABCDEF12-3456-7890-ABCD-EF1234567890'
    var_10 = module_1.UUID(var_9)
    var_11 = var_0.serialize(var_10)
    assert var_11 == 'abcdef12-3456-7890-abcd-ef1234567890'
    var_12 = var_0.is_native_type(var_4)
    assert var_12 is True
    var_13 = 'not a uuid'
    var_14 = var_0.is_native_type(var_13)
    assert var_14 is False
    var_15 = 123
    var_16 = var_0.is_native_type(var_15)
    assert var_16 is False



# Parsed testcases at query #20
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_9 = 'fe80::1'
    var_10 = '::1'
    var_11 = '2001:db8::1'
    var_12 = 'ff02::1'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'not an ip'
    var_15 = '256.256.256.256'
    var_16 = '192.168.1'
    var_17 = '192.168.1.1.1'
    var_18 = '192.168.1.256'
    var_19 = '2001:db8::1::'
    var_20 = 'gggg::1'
    var_21 = ''
    var_22 = None
    var_23 = 123
    var_24 = [var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23]
    var_25 = '999.999.999.999'
    var_26 = '300.300.300.300'
    var_27 = [var_25, var_26]



# Parsed testcases at query #21
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = '127.0.0.1'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_10 = '2001:db8:85a3::8a2e:370:7334'
    var_11 = 'fe80::1'
    var_12 = '::1'
    var_13 = '2001:0db8:0000:0000:0000:0000:0000:0001'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'not an ip'
    var_16 = '256.256.256.256'
    var_17 = '192.168.1'
    var_18 = '192.168.1.1.1'
    var_19 = '192.168.1.256'
    var_20 = '2001:db8:85a3::8a2e:370:7334:extra'
    var_21 = 'gggg::1'
    var_22 = '2001::db8::1'
    var_23 = [var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22]
    var_24 = '999.999.999.999'
    var_25 = '300.300.300.300'
    var_26 = [var_24, var_25]
    var_27 = module_1.IPv4Address(var_1)
    var_28 = var_0.is_native_type(var_27)
    assert var_28 is True
    var_29 = module_1.IPv6Address(var_12)
    var_30 = var_0.is_native_type(var_29)
    assert var_30 is True
    var_31 = var_0.is_native_type(var_1)
    assert var_31 is False
    var_32 = 123
    var_33 = var_0.is_native_type(var_32)
    assert var_33 is False
    var_34 = None
    var_35 = var_0.is_native_type(var_34)
    assert var_35 is False
    var_36 = var_0.serialize(var_34)
    assert var_36 is None
    var_37 = module_1.IPv4Address(var_1)
    var_38 = var_0.serialize(var_37)
    assert var_38 == '192.168.1.1'
    var_39 = module_1.IPv6Address(var_12)
    var_40 = var_0.serialize(var_39)
    assert var_40 == '::1'
    var_41 = 'not an ip'
    var_42 = var_0.serialize(var_41)



# Parsed testcases at query #22
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-567812345678'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = 'ABCDEFAB-1234-5678-1234-567812345678'
    var_5 = var_0.validate(var_4)
    var_6 = str(var_5)
    var_7 = 'invalid-uuid'
    var_8 = var_0.validate(var_7)
    var_9 = '12345678-1234-5678-1234-56781234567g'
    var_10 = var_0.validate(var_9)
    var_11 = '12345678-1234-5678-1234-5678123456789'
    var_12 = var_0.validate(var_11)
    var_13 = '12345678123456781234567812345678'
    var_14 = var_0.validate(var_13)
    var_15 = '12345678-1234-6234-1234-567812345678'
    var_16 = var_0.validate(var_15)
    var_17 = '12345678-1234-5234-6234-567812345678'
    var_18 = var_0.validate(var_17)



# Parsed testcases at query #23
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-1234-1234-123456789abc'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = 'ABCDEF00-1234-5678-9ABC-DEF012345678'
    var_5 = str(var_2)
    var_6 = 'not-a-uuid'
    var_7 = var_0.validate(var_6)
    var_8 = '12345678-1234-1234-1234-123456789abg'
    var_9 = var_0.validate(var_8)
    var_10 = '12345678-1234-6234-1234-123456789abc'
    var_11 = var_0.validate(var_10)
    var_12 = '12345678-1234-1234-7234-123456789abc'
    var_13 = var_0.validate(var_12)
    var_14 = '12345678-1234-1234-8234-123456789abc'
    var_15 = var_0.validate(var_14)
    var_16 = '12345678-1234-2234-8234-123456789abc'
    var_17 = var_0.validate(var_16)
    var_18 = '12345678-1234-3234-8234-123456789abc'
    var_19 = var_0.validate(var_18)
    var_20 = '12345678-1234-4234-8234-123456789abc'
    var_21 = var_0.validate(var_20)
    var_22 = '12345678-1234-5234-8234-123456789abc'
    var_23 = var_0.validate(var_22)



# Parsed testcases at query #24
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-1234-1234-123456789abc'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = 'a1b2c3d4-e5f6-1a2b-3c4d-5e6f7a8b9c0d'
    var_5 = var_0.validate(var_4)
    var_6 = str(var_5)
    var_7 = 'not-a-uuid'
    var_8 = var_0.validate(var_7)
    var_9 = '12345678-1234-1234-1234-123456789abg'
    var_10 = var_0.validate(var_9)
    var_11 = '12345678-1234-6234-1234-123456789abc'
    var_12 = var_0.validate(var_11)
    var_13 = '12345678-1234-1234-6234-123456789abc'
    var_14 = var_0.validate(var_13)
    var_15 = '12345678123412341234123456789abc'
    var_16 = var_0.validate(var_15)
    var_17 = '12345678-1234-1234-1234-123456789abcd'
    var_18 = var_0.validate(var_17)
    var_19 = '12345678-1234-1234-1234-123456789ab'
    var_20 = var_0.validate(var_19)
    var_21 = 'ABCDEF12-3456-789A-BCDE-F123456789AB'
    var_22 = var_0.validate(var_21)
    var_23 = str(var_22)



# Parsed testcases at query #25
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name@domain.co.uk'
    var_3 = 'user+tag@example.org'
    var_4 = 'user@sub.domain.com'
    var_5 = 'a@b.c'
    var_6 = 'user@example.io'
    var_7 = 'USER@EXAMPLE.COM'
    var_8 = 'user123@example.com'
    var_9 = 'first.last@company.co'
    var_10 = 'user@123.123.123.123'
    var_11 = '"email"@example.com'
    var_12 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10, var_11]
    var_13 = 'notanemail'
    var_14 = '@example.com'
    var_15 = 'user@'
    var_16 = 'user@.com'
    var_17 = 'user@example.'
    var_18 = 'user@example..com'
    var_19 = 'user@-example.com'
    var_20 = 'user@example-.com'
    var_21 = 'user name@example.com'
    var_22 = 'user@example com'
    var_23 = ''
    var_24 = None
    var_25 = 123
    var_26 = []
    var_27 = {}
    var_28 = [var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_24, var_25, var_26, var_27]
    var_29 = var_0.is_native_type(var_1)
    assert var_29 is False
    var_30 = var_0.is_native_type(var_24)
    assert var_30 is False
    var_31 = var_0.is_native_type(var_25)
    assert var_31 is False
    var_32 = var_0.serialize(var_1)
    assert var_32 == 'test@example.com'
    var_33 = var_0.serialize(var_24)
    assert var_33 is None
    var_34 = var_0.serialize(var_23)
    assert var_34 == ''



# Parsed testcases at query #26
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name@domain.co.uk'
    var_3 = 'user+tag@example.org'
    var_4 = 'user_name@sub.domain.com'
    var_5 = 'UPPERCASE@EXAMPLE.COM'
    var_6 = 'a@b.cd'
    var_7 = '"quoted"@example.com'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = 'notanemail'
    var_10 = '@example.com'
    var_11 = 'test@'
    var_12 = 'test@.com'
    var_13 = 'test@com'
    var_14 = 'test@example.'
    var_15 = 'test@example..com'
    var_16 = 'test @example.com'
    var_17 = 'test@example com'
    var_18 = ''
    var_19 = None
    var_20 = 123
    var_21 = []
    var_22 = {}
    var_23 = [var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22]
    var_24 = var_0.is_native_type(var_1)
    var_25 = var_0.is_native_type(var_19)
    var_26 = var_0.is_native_type(var_20)
    var_27 = var_0.serialize(var_1)
    assert var_27 == 'test@example.com'
    var_28 = var_0.serialize(var_19)
    assert var_28 is None



# Parsed testcases at query #27
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_9 = 'fe80::1'
    var_10 = '::1'
    var_11 = '2001:db8::1'
    var_12 = 'ff02::1'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'not an ip'
    var_15 = '256.256.256.256'
    var_16 = '192.168.1'
    var_17 = '192.168.1.1.1'
    var_18 = '192.168.1.256'
    var_19 = '2001:db8::1::'
    var_20 = 'gggg::1'
    var_21 = ''
    var_22 = None
    var_23 = 123
    var_24 = [var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23]
    var_25 = '999.999.999.999'
    var_26 = '300.300.300.300'
    var_27 = [var_25, var_26]
    var_28 = module_1.IPv4Address(var_1)
    var_29 = var_0.is_native_type(var_28)
    assert var_29 is True
    var_30 = module_1.IPv6Address(var_10)
    var_31 = var_0.is_native_type(var_30)
    assert var_31 is True
    var_32 = var_0.is_native_type(var_1)
    assert var_32 is False
    var_33 = var_0.is_native_type(var_23)
    assert var_33 is False
    var_34 = var_0.is_native_type(var_22)
    assert var_34 is False
    var_35 = module_1.IPv4Address(var_1)
    var_36 = module_1.IPv6Address(var_10)
    var_37 = var_0.serialize(var_35)
    assert var_37 == '192.168.1.1'
    var_38 = var_0.serialize(var_36)
    assert var_38 == '::1'
    var_39 = var_0.serialize(var_22)
    assert var_39 is None
    var_40 = 'not an ip'
    var_41 = var_0.serialize(var_40)



# Parsed testcases at query #28
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name@domain.co.uk'
    var_3 = 'user+tag@example.org'
    var_4 = 'user_name@sub.domain.com'
    var_5 = '123@numbers.com'
    var_6 = 'UPPERCASE@EXAMPLE.COM'
    var_7 = 'test.email@domain.io'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = 'notanemail'
    var_10 = '@no-local-part.com'
    var_11 = 'no-domain@'
    var_12 = 'spaces in@email.com'
    var_13 = 'invalid@.com'
    var_14 = '@@double.at.com'
    var_15 = 'missing@tld.'
    var_16 = 'invalid@-hyphen-start.com'
    var_17 = 'invalid@hyphen-end-.com'
    var_18 = [var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17]
    var_19 = ''
    var_20 = var_0.validate(var_19)
    var_21 = None
    var_22 = var_0.validate(var_21)
    var_23 = var_0.is_native_type(var_21)
    var_24 = None
    var_25 = var_0.is_native_type(var_24)
    var_26 = 123
    var_27 = var_0.is_native_type(var_26)
    var_28 = var_0.serialize(var_21)
    assert var_28 == 'test@example.com'
    var_29 = var_0.serialize(var_24)
    assert var_29 is None
    var_30 = ''
    var_31 = var_0.serialize(var_30)
    assert var_31 == ''



# Parsed testcases at query #29
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name@domain.co.uk'
    var_3 = 'user+tag@example.org'
    var_4 = 'user_name@sub.domain.com'
    var_5 = '123@numbers.com'
    var_6 = 'UPPERCASE@EXAMPLE.COM'
    var_7 = 'mixed.CASE@Example.Com'
    var_8 = '"quoted"@example.com'
    var_9 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8]
    var_10 = 'notanemail'
    var_11 = '@example.com'
    var_12 = 'test@'
    var_13 = 'test@.com'
    var_14 = 'test@com'
    var_15 = 'test@example.'
    var_16 = 'test@example..com'
    var_17 = 'test @example.com'
    var_18 = 'test@example com'
    var_19 = 'test@-example.com'
    var_20 = ''
    var_21 = None
    var_22 = 123
    var_23 = []
    var_24 = {}
    var_25 = [var_10, var_11, var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_24]
    var_26 = var_0.is_native_type(var_1)
    assert var_26 is False
    var_27 = var_0.is_native_type(var_21)
    assert var_27 is False
    var_28 = var_0.is_native_type(var_22)
    assert var_28 is False
    var_29 = var_0.serialize(var_1)
    assert var_29 == 'test@example.com'
    var_30 = var_0.serialize(var_21)
    assert var_30 is None
    var_31 = var_0.serialize(var_20)
    assert var_31 == ''



# Parsed testcases at query #30
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'http://example.com'
    var_2 = 'https://example.com'
    var_3 = 'http://example.com/path'
    var_4 = 'http://example.com/path?query=param'
    var_5 = 'http://example.com:8080'
    var_6 = 'http://user:pass@example.com'
    var_7 = 'ftp://example.com'
    var_8 = 'http://sub.example.com'
    var_9 = 'http://192.168.1.1'
    var_10 = 'http://localhost'
    var_11 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = 'example.com'
    var_13 = 'http://'
    var_14 = '://example.com'
    var_15 = ''
    var_16 = 'http:/example.com'
    var_17 = 'mailto:user@example.com'
    var_18 = [var_12, var_13, var_14, var_15, var_16, var_13, var_17]



# Parsed testcases at query #31
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name@domain.co.uk'
    var_3 = 'user+tag@example.org'
    var_4 = 'user@sub.domain.com'
    var_5 = 'a@b.cd'
    var_6 = 'user@example.io'
    var_7 = 'USER@EXAMPLE.COM'
    var_8 = 'user123@example.com'
    var_9 = 'first.last@domain.com'
    var_10 = '"quoted"@example.com'
    var_11 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = 'notanemail'
    var_13 = '@example.com'
    var_14 = 'user@'
    var_15 = 'user@.com'
    var_16 = 'user@example.'
    var_17 = 'user@example..com'
    var_18 = 'user @example.com'
    var_19 = 'user@exa mple.com'
    var_20 = 'user@-example.com'
    var_21 = ''
    var_22 = 'user@example.c'
    var_23 = [var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22]
    var_24 = var_0.is_native_type(var_1)
    assert var_24 is False
    var_25 = None
    var_26 = var_0.is_native_type(var_25)
    assert var_26 is False
    var_27 = 123
    var_28 = var_0.is_native_type(var_27)
    assert var_28 is False
    var_29 = var_0.serialize(var_1)
    assert var_29 == 'test@example.com'
    var_30 = var_0.serialize(var_25)
    assert var_30 is None
    var_31 = var_0.serialize(var_21)
    assert var_31 == ''



# Parsed testcases at query #32
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30
    var_5 = '09:45'
    var_6 = var_0.validate(var_5)
    var_7 = 9
    var_8 = 45
    var_9 = '23:59'
    var_10 = var_0.validate(var_9)
    var_11 = 23
    var_12 = 59
    var_13 = '00:00'
    var_14 = var_0.validate(var_13)
    var_15 = 0
    var_16 = '12:30:45'
    var_17 = var_0.validate(var_16)
    var_18 = '09:45:30'
    var_19 = var_0.validate(var_18)
    var_20 = '12:30:45.123456'
    var_21 = var_0.validate(var_20)
    var_22 = 123456
    var_23 = '09:45:30.500'
    var_24 = var_0.validate(var_23)
    var_25 = 500000
    var_26 = '12:30:45.12'
    var_27 = var_0.validate(var_26)
    var_28 = 120000
    var_29 = '25:30'
    var_30 = var_0.validate(var_29)
    var_31 = '12:60'
    var_32 = var_0.validate(var_31)
    var_33 = '12:30:60'
    var_34 = var_0.validate(var_33)
    var_35 = 'invalid-time'
    var_36 = var_0.validate(var_35)
    var_37 = '12:30:45.1234567'
    var_38 = var_0.validate(var_37)
    var_39 = '12:30:45.1234560'
    var_40 = var_0.validate(var_39)
    var_41 = var_0.is_native_type(var_39)
    assert var_41 is False
    var_42 = 123
    var_43 = var_0.is_native_type(var_42)
    assert var_43 is False
    var_44 = None
    var_45 = var_0.serialize(var_44)
    assert var_45 is None



# Parsed testcases at query #33
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'http://example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'http://example.com'
    var_3 = 'https://example.com'
    var_4 = var_0.validate(var_3)
    assert var_4 == 'https://example.com'
    var_5 = 'http://example.com/path'
    var_6 = var_0.validate(var_5)
    assert var_6 == 'http://example.com/path'
    var_7 = 'http://example.com:8080'
    var_8 = var_0.validate(var_7)
    assert var_8 == 'http://example.com:8080'
    var_9 = 'http://user:pass@example.com'
    var_10 = var_0.validate(var_9)
    assert var_10 == 'http://user:pass@example.com'
    var_11 = 'ftp://example.com'
    var_12 = var_0.validate(var_11)
    assert var_12 == 'ftp://example.com'
    var_13 = 'not-a-url'
    var_14 = var_0.validate(var_13)
    var_15 = 'http://'
    var_16 = var_0.validate(var_15)
    var_17 = '://example.com'
    var_18 = var_0.validate(var_17)
    var_19 = ''
    var_20 = var_0.validate(var_19)
    var_21 = None
    var_22 = var_0.validate(var_21)



# Parsed testcases at query #34
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'http://example.com'
    var_2 = 'https://example.com'
    var_3 = 'http://example.com/path'
    var_4 = 'http://example.com/path?query=param'
    var_5 = 'http://example.com:8080'
    var_6 = 'http://user:pass@example.com'
    var_7 = 'ftp://example.com'
    var_8 = 'http://sub.example.com'
    var_9 = 'http://192.168.1.1'
    var_10 = 'http://localhost'
    var_11 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = 'example.com'
    var_13 = 'http://'
    var_14 = '://example.com'
    var_15 = ''
    var_16 = 'http:/example.com'
    var_17 = 'mailto:user@example.com'
    var_18 = [var_12, var_13, var_14, var_15, var_16, var_13, var_17]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.URLFormat()
    var_1 = 'http://example.com'
    var_2 = 'https://example.com'
    var_3 = 'http://example.com/path'
    var_4 = 'http://example.com/path?query=param'
    var_5 = 'http://example.com:8080'
    var_6 = 'http://user:pass@example.com'
    var_7 = 'ftp://example.com'
    var_8 = 'http://sub.example.com'
    var_9 = 'http://192.168.1.1'
    var_10 = 'http://localhost'
    var_11 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = 'example.com'
    var_13 = 'http://'
    var_14 = '://example.com'
    var_15 = ''
    var_16 = 'http:/example.com'
    var_17 = 'mailto:user@example.com'
    var_18 = [var_12, var_13, var_14, var_15, var_16, var_13, var_17]
    var_19 = var_0.validate(var_1)
    var_20 = None
    var_21 = var_0.validate(var_20)



# Parsed testcases at query #2
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = 'user.name@domain.co.uk'
    var_3 = 'user+tag@example.org'
    var_4 = 'user@sub.domain.com'
    var_5 = 'a@b.cd'
    var_6 = 'user@123.456.789.123'
    var_7 = '"special@chars"@example.com'
    var_8 = 'UPPERCASE@EXAMPLE.COM'
    var_9 = 'lowercase@example.com'
    var_10 = 'MixedCase@Example.com'
    var_11 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8, var_9, var_10]
    var_12 = 'notanemail'
    var_13 = '@example.com'
    var_14 = 'user@'
    var_15 = 'user@.com'
    var_16 = 'user@domain.'
    var_17 = 'user@-domain.com'
    var_18 = 'user@domain-.com'
    var_19 = 'user name@example.com'
    var_20 = 'user@example..com'
    var_21 = ''
    var_22 = '   '
    var_23 = 'user@example.c'
    var_24 = 'user@.example.com'
    var_25 = [var_12, var_13, var_14, var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_24]
    var_26 = var_0.is_native_type(var_1)
    assert var_26 is False
    var_27 = None
    var_28 = var_0.is_native_type(var_27)
    assert var_28 is False
    var_29 = 123
    var_30 = var_0.is_native_type(var_29)
    assert var_30 is False
    var_31 = var_0.serialize(var_1)
    assert var_31 == 'test@example.com'
    var_32 = var_0.serialize(var_27)
    assert var_32 is None
    var_33 = var_0.serialize(var_21)
    assert var_33 == ''



# Parsed testcases at query #3
#--------------------------


import typesystem.formats as module_0
import datetime as module_1
import uuid as module_2
import ipaddress as module_3

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2023
    var_4 = 12
    var_5 = 25
    var_6 = module_0.TimeFormat()
    var_7 = var_6.serialize(var_1)
    assert var_7 is None
    var_8 = 14
    var_9 = 30
    var_10 = 45
    var_11 = 123456
    var_12 = module_0.DateTimeFormat()
    var_13 = var_12.serialize(var_1)
    assert var_13 is None
    var_14 = 5
    var_15 = module_1.timedelta()
    var_16 = module_0.UUIDFormat()
    var_17 = var_16.serialize(var_1)
    assert var_17 is None
    var_18 = '12345678-1234-5678-1234-567812345678'
    var_19 = module_2.UUID(var_18)
    var_20 = var_16.serialize(var_19)
    assert var_20 == '12345678-1234-5678-1234-567812345678'
    var_21 = module_0.EmailFormat()
    var_22 = var_21.serialize(var_1)
    assert var_22 is None
    var_23 = 'test@example.com'
    var_24 = var_21.serialize(var_23)
    assert var_24 == 'test@example.com'
    var_25 = module_0.IPAddressFormat()
    var_26 = var_25.serialize(var_1)
    assert var_26 is None
    var_27 = '192.168.1.1'
    var_28 = module_3.IPv4Address(var_27)
    var_29 = var_25.serialize(var_28)
    assert var_29 == '192.168.1.1'
    var_30 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_31 = module_3.IPv6Address(var_30)
    var_32 = var_25.serialize(var_31)
    assert var_32 == '2001:db8:85a3::8a2e:370:7334'
    var_33 = module_0.URLFormat()
    var_34 = var_33.serialize(var_1)
    assert var_34 is None
    var_35 = 'https://example.com/path'
    var_36 = var_33.serialize(var_35)
    assert var_36 == 'https://example.com/path'



# Parsed testcases at query #4
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.TimeFormat()
    var_1 = '12:30'
    var_2 = var_0.validate(var_1)
    var_3 = 12
    var_4 = 30
    var_5 = '09:45'
    var_6 = var_0.validate(var_5)
    var_7 = 9
    var_8 = 45
    var_9 = '23:59'
    var_10 = var_0.validate(var_9)
    var_11 = 23
    var_12 = 59
    var_13 = '00:00'
    var_14 = var_0.validate(var_13)
    var_15 = 0
    var_16 = '12:30:45'
    var_17 = var_0.validate(var_16)
    var_18 = '09:45:30'
    var_19 = var_0.validate(var_18)
    var_20 = '12:30:45.123456'
    var_21 = var_0.validate(var_20)
    var_22 = 123456
    var_23 = '09:45:30.500'
    var_24 = var_0.validate(var_23)
    var_25 = 500000
    var_26 = '12:30:45.1'
    var_27 = var_0.validate(var_26)
    var_28 = 100000
    var_29 = '12:30:45.123'
    var_30 = var_0.validate(var_29)
    var_31 = 123000
    var_32 = '12:30:45.12'
    var_33 = var_0.validate(var_32)
    var_34 = 120000
    var_35 = '25:00'
    var_36 = var_0.validate(var_35)
    var_37 = '12:60'
    var_38 = var_0.validate(var_37)
    var_39 = '12:30:60'
    var_40 = var_0.validate(var_39)
    var_41 = 'not-a-time'
    var_42 = var_0.validate(var_41)
    var_43 = '12:30:45.1234567'
    var_44 = var_0.validate(var_43)
    var_45 = '12:30:45.'
    var_46 = var_0.validate(var_45)
    var_47 = var_0.validate(var_16)
    var_48 = None
    var_49 = '0:0'
    var_50 = var_0.validate(var_49)
    var_51 = '1:2:3'
    var_52 = var_0.validate(var_51)
    var_53 = 1
    var_54 = 2
    var_55 = 3
    var_56 = '1:2:3.4'
    var_57 = var_0.validate(var_56)
    var_58 = 400000



# Parsed testcases at query #5
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_9 = 'fe80::1'
    var_10 = '::1'
    var_11 = '2001:db8::1'
    var_12 = 'ff02::1'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'not an ip'
    var_15 = '192.168.1'
    var_16 = '192.168.1.256'
    var_17 = '192.168.1.1.1'
    var_18 = '2001:db8:85a3::8a2e:0370:7334:extra'
    var_19 = '2001::db8::1'
    var_20 = [var_14, var_15, var_16, var_17, var_18, var_19]
    var_21 = '999.999.999.999'
    var_22 = '256.256.256.256'
    var_23 = [var_21, var_22]
    var_24 = module_1.IPv4Address(var_1)
    var_25 = var_0.is_native_type(var_24)
    assert var_25 is True
    var_26 = module_1.IPv6Address(var_10)
    var_27 = var_0.is_native_type(var_26)
    assert var_27 is True
    var_28 = var_0.is_native_type(var_1)
    assert var_28 is False
    var_29 = 123
    var_30 = var_0.is_native_type(var_29)
    assert var_30 is False
    var_31 = None
    var_32 = var_0.is_native_type(var_31)
    assert var_32 is False
    var_33 = module_1.IPv4Address(var_1)
    var_34 = module_1.IPv6Address(var_10)
    var_35 = var_0.serialize(var_33)
    assert var_35 == '192.168.1.1'
    var_36 = var_0.serialize(var_34)
    assert var_36 == '::1'
    var_37 = var_0.serialize(var_31)
    assert var_37 is None
    var_38 = 'not an ip'
    var_39 = var_0.serialize(var_38)



# Parsed testcases at query #6
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '0.0.0.0'
    var_6 = '255.255.255.255'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_9 = '::1'
    var_10 = '2001:db8::1'
    var_11 = 'fe80::1'
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = 'not_an_ip'
    var_14 = '192.168.1'
    var_15 = '192.168.1.256'
    var_16 = '2001:db8:xyz::1'
    var_17 = ''
    var_18 = [var_13, var_14, var_15, var_16, var_17]
    var_19 = '999.999.999.999'
    var_20 = '2001:db8::1::1'
    var_21 = [var_19, var_15, var_20]
    var_22 = module_1.IPv4Address(var_1)
    var_23 = var_0.is_native_type(var_22)
    assert var_23 is True
    var_24 = module_1.IPv6Address(var_9)
    var_25 = var_0.is_native_type(var_24)
    assert var_25 is True
    var_26 = var_0.is_native_type(var_1)
    assert var_26 is False
    var_27 = 123
    var_28 = var_0.is_native_type(var_27)
    assert var_28 is False
    var_29 = None
    var_30 = var_0.is_native_type(var_29)
    assert var_30 is False
    var_31 = var_0.serialize(var_29)
    assert var_31 is None
    var_32 = module_1.IPv4Address(var_1)
    var_33 = var_0.serialize(var_32)
    assert var_33 == '192.168.1.1'
    var_34 = module_1.IPv6Address(var_9)
    var_35 = var_0.serialize(var_34)
    assert var_35 == '::1'



# Parsed testcases at query #7
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-2-5'
    var_4 = var_0.validate(var_3)
    var_5 = '2023/12/25'
    var_6 = var_0.validate(var_5)
    var_7 = '2023-12-25T10:30:00'
    var_8 = var_0.validate(var_7)
    var_9 = '2023-02-30'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-04-31'
    var_12 = var_0.validate(var_11)
    var_13 = '2023-13-01'
    var_14 = var_0.validate(var_13)
    var_15 = '2023-12-00'
    var_16 = var_0.validate(var_15)
    var_17 = '2024-02-29'
    var_18 = var_0.validate(var_17)
    var_19 = '2023-02-29'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-01-09'
    var_22 = var_0.validate(var_21)
    var_23 = '0001-01-01'
    var_24 = var_0.validate(var_23)
    var_25 = '9999-12-31'
    var_26 = var_0.validate(var_25)



# Parsed testcases at query #8
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2023
    var_4 = 12
    var_5 = 25
    var_6 = 2020
    var_7 = 2
    var_8 = 29
    var_9 = 1
    var_10 = 9999
    var_11 = 31
    var_12 = '2023-12-25'
    var_13 = var_0.serialize(var_12)
    var_14 = 123
    var_15 = var_0.serialize(var_14)
    var_16 = var_0.serialize(var_14)



# Parsed testcases at query #9
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
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = module_1.timedelta()
    var_11 = -2
    var_12 = module_1.timedelta()
    var_13 = module_1.timedelta()
    var_14 = 0
    var_15 = 123
    var_16 = 23
    var_17 = 59
    var_18 = 999999



# Parsed testcases at query #10
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-1-5'
    var_4 = var_0.validate(var_3)
    var_5 = '2023-12'
    var_6 = var_0.validate(var_5)
    var_7 = '2023/12/25'
    var_8 = var_0.validate(var_7)
    var_9 = '2023-12-25T10:30:00'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-13-25'
    var_12 = var_0.validate(var_11)
    var_13 = '2023-02-30'
    var_14 = var_0.validate(var_13)
    var_15 = '2023-02-29'
    var_16 = var_0.validate(var_15)
    var_17 = '2024-02-29'
    var_18 = var_0.validate(var_17)
    var_19 = '2023-12-00'
    var_20 = var_0.validate(var_19)
    var_21 = '2023-00-25'
    var_22 = var_0.validate(var_21)
    var_23 = '2023-01-01'
    var_24 = var_0.validate(var_23)
    var_25 = 12345
    var_26 = var_0.validate(var_25)
    var_27 = ''
    var_28 = var_0.validate(var_27)
    var_29 = '   '
    var_30 = var_0.validate(var_29)



# Parsed testcases at query #11
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-1234-1234-123456789abc'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = 'a1b2c3d4-e5f6-1a2b-3c4d-5e6f7a8b9c0d'
    var_5 = var_0.validate(var_4)
    var_6 = str(var_5)
    var_7 = '12345678-1234-1234-1234-123456789ab'
    var_8 = var_0.validate(var_7)
    var_9 = 'g1234567-1234-1234-1234-123456789abc'
    var_10 = var_0.validate(var_9)
    var_11 = '12345678-1234-6234-1234-123456789abc'
    var_12 = var_0.validate(var_11)
    var_13 = '12345678-1234-1234-c234-123456789abc'
    var_14 = var_0.validate(var_13)
    var_15 = ''
    var_16 = var_0.validate(var_15)
    var_17 = 12345
    var_18 = var_0.validate(var_17)



# Parsed testcases at query #12
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-01-15T10:30:45'
    var_2 = 2023
    var_3 = 1
    var_4 = 15
    var_5 = 10
    var_6 = 30
    var_7 = 45
    var_8 = '2023-01-15 10:30:45'
    var_9 = '2023-01-15T10:30'
    var_10 = '2023-01-15T10:30:45.123'
    var_11 = 123000
    var_12 = '2023-01-15T10:30:45.123456'
    var_13 = 123456
    var_14 = '2023-01-15T10:30:45.12'
    var_15 = 120000
    var_16 = '2023-01-15T10:30:45Z'
    var_17 = '2023-01-15T10:30:45+05:30'
    var_18 = 5
    var_19 = module_1.timedelta()
    var_20 = '2023-01-15T10:30:45-05:30'
    var_21 = -5
    var_22 = -30
    var_23 = module_1.timedelta()
    var_24 = '2023-01-15T10:30:45+0530'
    var_25 = module_1.timedelta()
    var_26 = '2023-01-15T10:30:45-0530'
    var_27 = -5
    var_28 = -30
    var_29 = module_1.timedelta()
    var_30 = '2023-01-15T10:30:45+05'
    var_31 = module_1.timedelta()
    var_32 = '2023-01-15'
    var_33 = '10:30:45'
    var_34 = '2023/01/15T10:30:45'
    var_35 = '2023-01-15T10:30:45.1234567'
    var_36 = '2023-01-15T10:30:45+'
    var_37 = '2023-01-15T10:30:45+5:30'
    var_38 = 'not-a-datetime'
    var_39 = [var_32, var_33, var_34, var_35, var_36, var_37, var_38]
    var_40 = '2023-13-01T10:30:45'
    var_41 = '2023-01-32T10:30:45'
    var_42 = '2023-01-15T25:30:45'
    var_43 = '2023-01-15T10:70:45'
    var_44 = '2023-02-30T10:30:45'
    var_45 = '2023-01-15T10:30:61'
    var_46 = [var_40, var_41, var_42, var_43, var_44, var_45]
    var_47 = var_0.is_native_type(var_1)
    assert var_47 is False
    var_48 = None
    var_49 = var_0.is_native_type(var_48)
    assert var_49 is False
    var_50 = 123
    var_51 = var_0.is_native_type(var_50)
    assert var_51 is False
    var_52 = var_0.serialize(var_48)
    assert var_52 is None



# Parsed testcases at query #13
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = '2023-10-05T14:30:45'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-10-05 14:30:45'
    var_4 = var_0.validate(var_3)
    var_5 = '2023-10-05T14:30:45.123456'
    var_6 = var_0.validate(var_5)
    var_7 = '2023-10-05T14:30:45.123'
    var_8 = var_0.validate(var_7)
    var_9 = '2023-10-05T14:30:45Z'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-10-05T14:30:45+05:30'
    var_12 = var_0.validate(var_11)
    var_13 = None
    var_14 = 5
    var_15 = 30
    var_16 = module_1.timedelta()
    var_17 = '2023-10-05T14:30:45-03:00'
    var_18 = var_0.validate(var_17)
    var_19 = -3
    var_20 = module_1.timedelta()
    var_21 = '2023-10-05T14:30:45+0530'
    var_22 = var_0.validate(var_21)
    var_23 = module_1.timedelta()
    var_24 = '2023-10-05T14:30:45+05'
    var_25 = var_0.validate(var_24)
    var_26 = module_1.timedelta()
    var_27 = '2023-10-05'
    var_28 = var_0.validate(var_27)
    var_29 = '10-05-2023T14:30:45'
    var_30 = var_0.validate(var_29)
    var_31 = '2023-13-05T14:30:45'
    var_32 = var_0.validate(var_31)
    var_33 = '2023-10-05T25:30:45'
    var_34 = var_0.validate(var_33)
    var_35 = '2023-02-30T14:30:45'
    var_36 = var_0.validate(var_35)
    var_37 = '2023-10-05T14:30:45+5:30'
    var_38 = var_0.validate(var_37)
    var_39 = ''
    var_40 = var_0.validate(var_39)



# Parsed testcases at query #14
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-5678-1234-567812345678'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
    var_5 = var_0.validate(var_4)
    var_6 = str(var_5)
    var_7 = 'not-a-uuid'
    var_8 = var_0.validate(var_7)
    var_9 = '12345678-1234-5678-1234-56781234567g'
    var_10 = var_0.validate(var_9)
    var_11 = '12345678-1234-5678-1234-5678123456789'
    var_12 = var_0.validate(var_11)
    var_13 = '12345678123456781234567812345678'
    var_14 = var_0.validate(var_13)
    var_15 = '12345678-1234-6234-1234-567812345678'
    var_16 = var_0.validate(var_15)
    var_17 = '12345678-1234-5234-6234-567812345678'
    var_18 = var_0.validate(var_17)



# Parsed testcases at query #15
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = '127.0.0.1'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_10 = '2001:db8:85a3::8a2e:370:7334'
    var_11 = 'fe80::1'
    var_12 = '::1'
    var_13 = '2001:0db8:0000:0000:0000:0000:0000:0001'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'not an ip'
    var_16 = '256.256.256.256'
    var_17 = '192.168.1'
    var_18 = '192.168.1.1.1'
    var_19 = '192.168.1.256'
    var_20 = '2001:db8:85a3::8a2e:370:7334:extra'
    var_21 = 'gggg::1'
    var_22 = '2001:db8:85a3:8a2e:370:7334'
    var_23 = [var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22]
    var_24 = '999.999.999.999'
    var_25 = '300.300.300.300'
    var_26 = [var_24, var_25]
    var_27 = module_1.IPv4Address(var_1)
    var_28 = var_0.is_native_type(var_27)
    assert var_28 is True
    var_29 = module_1.IPv6Address(var_12)
    var_30 = var_0.is_native_type(var_29)
    assert var_30 is True
    var_31 = var_0.is_native_type(var_1)
    assert var_31 is False
    var_32 = 123
    var_33 = var_0.is_native_type(var_32)
    assert var_33 is False
    var_34 = None
    var_35 = var_0.is_native_type(var_34)
    assert var_35 is False
    var_36 = module_1.IPv4Address(var_1)
    var_37 = var_0.serialize(var_36)
    assert var_37 == '192.168.1.1'
    var_38 = module_1.IPv6Address(var_12)
    var_39 = var_0.serialize(var_38)
    assert var_39 == '::1'
    var_40 = var_0.serialize(var_34)
    assert var_40 is None
    var_41 = 'not an ip'
    var_42 = var_0.serialize(var_41)



# Parsed testcases at query #16
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = '127.0.0.1'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_10 = 'fe80::1'
    var_11 = '::1'
    var_12 = '2001:db8::1'
    var_13 = 'ff02::1'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'not an ip'
    var_16 = '256.256.256.256'
    var_17 = '192.168.1'
    var_18 = '192.168.1.1.1'
    var_19 = '192.168.1.256'
    var_20 = '2001:db8::1::'
    var_21 = 'gggg::1'
    var_22 = ''
    var_23 = None
    var_24 = 12345
    var_25 = [var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_24]
    var_26 = '999.999.999.999'
    var_27 = '300.300.300.300'
    var_28 = [var_26, var_27]
    var_29 = module_1.IPv4Address(var_1)
    var_30 = var_0.is_native_type(var_29)
    assert var_30 is True
    var_31 = module_1.IPv6Address(var_11)
    var_32 = var_0.is_native_type(var_31)
    assert var_32 is True
    var_33 = var_0.is_native_type(var_1)
    assert var_33 is False
    var_34 = 123
    var_35 = var_0.is_native_type(var_34)
    assert var_35 is False
    var_36 = var_0.is_native_type(var_23)
    assert var_36 is False
    var_37 = var_0.serialize(var_23)
    assert var_37 is None
    var_38 = module_1.IPv4Address(var_1)
    var_39 = var_0.serialize(var_38)
    assert var_39 == '192.168.1.1'
    var_40 = module_1.IPv6Address(var_11)
    var_41 = var_0.serialize(var_40)
    assert var_41 == '::1'



# Parsed testcases at query #17
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
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = module_1.timedelta()
    var_11 = -2
    var_12 = module_1.timedelta()
    var_13 = module_1.timedelta()
    var_14 = 123
    var_15 = 0
    var_16 = 23
    var_17 = 59
    var_18 = 999999



# Parsed testcases at query #18
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = '127.0.0.1'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_10 = '2001:db8:85a3::8a2e:370:7334'
    var_11 = 'fe80::1'
    var_12 = '::1'
    var_13 = '2001:0db8:0000:0000:0000:0000:0000:0001'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'not_an_ip'
    var_16 = '256.256.256.256'
    var_17 = '192.168.1'
    var_18 = '192.168.1.1.1'
    var_19 = '2001:db8:85a3::8a2e:370:7334:extra'
    var_20 = '192.168.1.256'
    var_21 = 'fe80::1::'
    var_22 = ''
    var_23 = [var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22]
    var_24 = '999.999.999.999'
    var_25 = [var_24]
    var_26 = module_1.IPv4Address(var_1)
    var_27 = module_1.IPv6Address(var_12)
    var_28 = var_0.is_native_type(var_26)
    assert var_28 is True
    var_29 = var_0.is_native_type(var_27)
    assert var_29 is True
    var_30 = var_0.serialize(var_26)
    assert var_30 == '192.168.1.1'
    var_31 = var_0.serialize(var_27)
    assert var_31 == '::1'
    var_32 = None
    var_33 = var_0.serialize(var_32)
    assert var_33 is None
    var_34 = 'invalid'
    var_35 = var_0.validate(var_34)



# Parsed testcases at query #19
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = '127.0.0.1'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_10 = '2001:db8:85a3::8a2e:370:7334'
    var_11 = 'fe80::1'
    var_12 = '::1'
    var_13 = '2001:0db8:0000:0000:0000:0000:0000:0001'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'not an ip'
    var_16 = '256.256.256.256'
    var_17 = '192.168.1'
    var_18 = '192.168.1.1.1'
    var_19 = '192.168.1.256'
    var_20 = '2001:db8:85a3::8a2e:370:7334:extra'
    var_21 = 'gggg::1'
    var_22 = ''
    var_23 = None
    var_24 = 12345
    var_25 = [var_1]
    var_26 = [var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_24, var_25]
    var_27 = '999.999.999.999'
    var_28 = '300.300.300.300'
    var_29 = [var_27, var_28]



# Parsed testcases at query #20
#--------------------------


import typesystem.formats as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.DateTimeFormat()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = 2023
    var_4 = 5
    var_5 = 15
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = module_1.timedelta()
    var_11 = -3
    var_12 = module_1.timedelta()
    var_13 = module_1.timedelta()
    var_14 = 0
    var_15 = 123



# Parsed testcases at query #21
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-1234-1234-123456789abc'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = '12345678-1234-6234-1234-123456789abc'
    var_5 = var_0.validate(var_4)
    var_6 = 'not-a-uuid'
    var_7 = var_0.validate(var_6)
    var_8 = '12345678-1234-1234-1234-123456789abg'
    var_9 = var_0.validate(var_8)
    var_10 = '12345678-1234-1234-1234-123456789ABC'
    var_11 = var_0.validate(var_10)
    var_12 = str(var_11)
    var_13 = '12345678-1234-1234-8123-123456789abc'
    var_14 = var_0.validate(var_13)
    var_15 = '12345678-1234-3123-8123-123456789abc'
    var_16 = var_0.validate(var_15)
    var_17 = '12345678-1234-4123-8123-123456789abc'
    var_18 = var_0.validate(var_17)
    var_19 = '12345678-1234-5123-8123-123456789abc'
    var_20 = var_0.validate(var_19)



# Parsed testcases at query #22
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_9 = 'fe80::1'
    var_10 = '::1'
    var_11 = '2001:db8::1'
    var_12 = 'ff02::1'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'not an ip'
    var_15 = '256.256.256.256'
    var_16 = '192.168.1'
    var_17 = '192.168.1.1.1'
    var_18 = '2001:db8::1::'
    var_19 = 'gggg::1'
    var_20 = [var_14, var_15, var_16, var_17, var_18, var_19]
    var_21 = '999.999.999.999'
    var_22 = [var_21]
    var_23 = module_1.IPv4Address(var_1)
    var_24 = var_0.is_native_type(var_23)
    assert var_24 is True
    var_25 = module_1.IPv6Address(var_10)
    var_26 = var_0.is_native_type(var_25)
    assert var_26 is True
    var_27 = var_0.is_native_type(var_1)
    assert var_27 is False
    var_28 = 123
    var_29 = var_0.is_native_type(var_28)
    assert var_29 is False
    var_30 = None
    var_31 = var_0.is_native_type(var_30)
    assert var_31 is False
    var_32 = var_0.serialize(var_30)
    assert var_32 is None
    var_33 = module_1.IPv4Address(var_1)
    var_34 = var_0.serialize(var_33)
    assert var_34 == '192.168.1.1'
    var_35 = module_1.IPv6Address(var_10)
    var_36 = var_0.serialize(var_35)
    assert var_36 == '::1'
    var_37 = 'not an ip'
    var_38 = var_0.serialize(var_37)



# Parsed testcases at query #23
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = '127.0.0.1'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_10 = '2001:db8:85a3::8a2e:370:7334'
    var_11 = 'fe80::1'
    var_12 = '::1'
    var_13 = '2001:0db8:0000:0000:0000:0000:0000:0001'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'not an ip'
    var_16 = '256.256.256.256'
    var_17 = '192.168.1'
    var_18 = '192.168.1.1.1'
    var_19 = '192.168.1.256'
    var_20 = '2001:db8:85a3::8a2e:370:7334:extra'
    var_21 = 'gggg::1'
    var_22 = ''
    var_23 = None
    var_24 = 12345
    var_25 = [var_1]
    var_26 = [var_15, var_16, var_17, var_18, var_19, var_20, var_21, var_22, var_23, var_24, var_25]
    var_27 = '999.999.999.999'
    var_28 = '300.300.300.300'
    var_29 = [var_27, var_28]



# Parsed testcases at query #24
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.EmailFormat()
    var_1 = 'test@example.com'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'test@example.com'
    var_3 = 'user.name@domain.co.uk'
    var_4 = var_0.validate(var_3)
    assert var_4 == 'user.name@domain.co.uk'
    var_5 = 'user+tag@example.org'
    var_6 = var_0.validate(var_5)
    assert var_6 == 'user+tag@example.org'
    var_7 = 'user_name@sub.domain.com'
    var_8 = var_0.validate(var_7)
    assert var_8 == 'user_name@sub.domain.com'
    var_9 = '123@numbers.com'
    var_10 = var_0.validate(var_9)
    assert var_10 == '123@numbers.com'
    var_11 = 'UPPERCASE@EXAMPLE.COM'
    var_12 = var_0.validate(var_11)
    assert var_12 == 'UPPERCASE@EXAMPLE.COM'
    var_13 = 'a@b.cd'
    var_14 = var_0.validate(var_13)
    assert var_14 == 'a@b.cd'
    var_15 = 'invalid-email'
    var_16 = var_0.validate(var_15)
    var_17 = 'missing@domain'
    var_18 = var_0.validate(var_17)
    var_19 = '@nodomain.com'
    var_20 = var_0.validate(var_19)
    var_21 = 'noat.com'
    var_22 = var_0.validate(var_21)
    var_23 = 'spaces in@email.com'
    var_24 = var_0.validate(var_23)
    var_25 = ''
    var_26 = var_0.validate(var_25)
    var_27 = 'invalid'
    var_28 = var_0.validate(var_27)



# Parsed testcases at query #25
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '255.255.255.255'
    var_4 = '0.0.0.0'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_7 = '::1'
    var_8 = '2001:db8::1'
    var_9 = [var_6, var_7, var_8]
    var_10 = 'not_an_ip'
    var_11 = '192.168.1'
    var_12 = '192.168.1.256'
    var_13 = '2001::db8::1'
    var_14 = [var_10, var_11, var_12, var_13]
    var_15 = '999.999.999.999'
    var_16 = '2001:db8:xyz::1'
    var_17 = [var_15, var_16]
    var_18 = module_1.IPv4Address(var_1)
    var_19 = var_0.is_native_type(var_18)
    assert var_19 is True
    var_20 = module_1.IPv6Address(var_7)
    var_21 = var_0.is_native_type(var_20)
    assert var_21 is True
    var_22 = var_0.is_native_type(var_1)
    assert var_22 is False
    var_23 = 123
    var_24 = var_0.is_native_type(var_23)
    assert var_24 is False
    var_25 = None
    var_26 = var_0.serialize(var_25)
    assert var_26 is None
    var_27 = module_1.IPv4Address(var_1)
    var_28 = var_0.serialize(var_27)
    assert var_28 == '192.168.1.1'
    var_29 = module_1.IPv6Address(var_7)
    var_30 = var_0.serialize(var_29)
    assert var_30 == '::1'



# Parsed testcases at query #26
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
    var_5 = 5
    var_6 = 14
    var_7 = 30
    var_8 = 45
    var_9 = 123456
    var_10 = module_1.timedelta()
    var_11 = -2
    var_12 = module_1.timedelta()
    var_13 = module_1.timedelta()
    var_14 = 0
    var_15 = 123



# Parsed testcases at query #27
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.DateFormat()
    var_1 = '2023-12-25'
    var_2 = var_0.validate(var_1)
    var_3 = '2023-2-5'
    var_4 = var_0.validate(var_3)
    var_5 = '2023/12/25'
    var_6 = var_0.validate(var_5)
    var_7 = '2023-02-30'
    var_8 = var_0.validate(var_7)
    var_9 = '2023-13-01'
    var_10 = var_0.validate(var_9)
    var_11 = '2023-04-31'
    var_12 = var_0.validate(var_11)
    var_13 = '2024-02-29'
    var_14 = var_0.validate(var_13)
    var_15 = '2023-02-29'
    var_16 = var_0.validate(var_15)
    var_17 = '0001-01-01'
    var_18 = var_0.validate(var_17)
    var_19 = ' 2023-12-25 '
    var_20 = var_0.validate(var_19)
    var_21 = ''
    var_22 = var_0.validate(var_21)
    var_23 = None
    var_24 = var_0.validate(var_23)



# Parsed testcases at query #28
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.UUIDFormat()
    var_1 = '12345678-1234-1234-1234-123456789abc'
    var_2 = var_0.validate(var_1)
    var_3 = str(var_2)
    var_4 = 'not-a-uuid'
    var_5 = var_0.validate(var_4)
    var_6 = '12345678-1234-6234-1234-123456789abc'
    var_7 = var_0.validate(var_6)
    var_8 = '12345678-1234-1234-c234-123456789abc'
    var_9 = var_0.validate(var_8)
    var_10 = '12345678-1234-1234-1234-123456789ABC'
    var_11 = var_0.validate(var_10)
    var_12 = str(var_11)
    var_13 = '12345678-1234-1234-1234-123456789AbC'
    var_14 = var_0.validate(var_13)
    var_15 = str(var_14)



# Parsed testcases at query #29
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '0.0.0.0'
    var_6 = '255.255.255.255'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_9 = '::1'
    var_10 = '2001:db8::1'
    var_11 = 'fe80::1'
    var_12 = [var_8, var_9, var_10, var_11]
    var_13 = 'not_an_ip'
    var_14 = '192.168.1'
    var_15 = '192.168.1.256'
    var_16 = '192.168.1.1.1'
    var_17 = '2001:db8:xyz::1'
    var_18 = [var_13, var_14, var_15, var_16, var_17]
    var_19 = '999.999.999.999'
    var_20 = '300.300.300.300'
    var_21 = [var_19, var_20]
    var_22 = module_1.IPv4Address(var_1)
    var_23 = var_0.is_native_type(var_22)
    assert var_23 is True
    var_24 = module_1.IPv6Address(var_9)
    var_25 = var_0.is_native_type(var_24)
    assert var_25 is True
    var_26 = var_0.is_native_type(var_1)
    assert var_26 is False
    var_27 = 123
    var_28 = var_0.is_native_type(var_27)
    assert var_28 is False
    var_29 = None
    var_30 = var_0.is_native_type(var_29)
    assert var_30 is False
    var_31 = var_0.serialize(var_29)
    assert var_31 is None
    var_32 = module_1.IPv4Address(var_1)
    var_33 = var_0.serialize(var_32)
    assert var_33 == '192.168.1.1'
    var_34 = module_1.IPv6Address(var_9)
    var_35 = var_0.serialize(var_34)
    assert var_35 == '::1'
    var_36 = 'not_an_ip_object'
    var_37 = var_0.serialize(var_36)



# Parsed testcases at query #30
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '0.0.0.0'
    var_6 = '255.255.255.255'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_9 = '::1'
    var_10 = '2001:db8::1'
    var_11 = [var_8, var_9, var_10]
    var_12 = 'not_an_ip'
    var_13 = '192.168.1'
    var_14 = '192.168.1.256'
    var_15 = '2001:db8:xyz::1'
    var_16 = [var_12, var_13, var_14, var_15]
    var_17 = '999.999.999.999'
    var_18 = [var_14, var_17]
    var_19 = module_1.IPv4Address(var_1)
    var_20 = var_0.is_native_type(var_19)
    var_21 = module_1.IPv6Address(var_9)
    var_22 = var_0.is_native_type(var_21)
    var_23 = var_0.serialize(var_19)
    assert var_23 == '192.168.1.1'
    var_24 = var_0.serialize(var_21)
    assert var_24 == '::1'
    var_25 = None
    var_26 = var_0.serialize(var_25)
    assert var_26 is None



# Parsed testcases at query #31
#--------------------------


import typesystem.formats as module_0

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_8 = '::1'
    var_9 = '2001:db8::1'
    var_10 = [var_7, var_8, var_9]
    var_11 = 'not_an_ip'
    var_12 = '192.168.1'
    var_13 = '192.168.1.256'
    var_14 = '2001::db8::1'
    var_15 = [var_11, var_12, var_13, var_14]
    var_16 = '999.999.999.999'
    var_17 = '2001:db8:xyz::1'
    var_18 = [var_16, var_17]



# Parsed testcases at query #32
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = var_0.validate(var_1)
    var_3 = module_1.IPv4Address(var_1)
    var_4 = '0.0.0.0'
    var_5 = var_0.validate(var_4)
    var_6 = module_1.IPv4Address(var_4)
    var_7 = '255.255.255.255'
    var_8 = var_0.validate(var_7)
    var_9 = module_1.IPv4Address(var_7)
    var_10 = '127.0.0.1'
    var_11 = var_0.validate(var_10)
    var_12 = module_1.IPv4Address(var_10)
    var_13 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_14 = var_0.validate(var_13)
    var_15 = module_1.IPv6Address(var_13)
    var_16 = 'fe80::1'
    var_17 = var_0.validate(var_16)
    var_18 = module_1.IPv6Address(var_16)
    var_19 = '::1'
    var_20 = var_0.validate(var_19)
    var_21 = module_1.IPv6Address(var_19)
    var_22 = '2001:db8::1'
    var_23 = var_0.validate(var_22)
    var_24 = module_1.IPv6Address(var_22)
    var_25 = 'not-an-ip'
    var_26 = var_0.validate(var_25)
    var_27 = '256.256.256.256'
    var_28 = var_0.validate(var_27)
    var_29 = '192.168.1'
    var_30 = var_0.validate(var_29)
    var_31 = '999.999.999.999'
    var_32 = var_0.validate(var_31)
    var_33 = '192.168.1.256'
    var_34 = var_0.validate(var_33)
    var_35 = ''
    var_36 = var_0.validate(var_35)
    var_37 = None
    var_38 = var_0.validate(var_37)



# Parsed testcases at query #33
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = [var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_9 = 'fe80::1'
    var_10 = '::1'
    var_11 = '2001:db8::1'
    var_12 = 'ff02::1'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'not an ip'
    var_15 = '192.168.1'
    var_16 = '192.168.1.256'
    var_17 = '192.168.1.1.1'
    var_18 = '2001:db8:85a3::8a2e:0370:7334:extra'
    var_19 = 'gggg::1'
    var_20 = [var_14, var_15, var_16, var_17, var_18, var_19]
    var_21 = '999.999.999.999'
    var_22 = '256.256.256.256'
    var_23 = [var_21, var_22]
    var_24 = module_1.IPv4Address(var_1)
    var_25 = var_0.is_native_type(var_24)
    assert var_25 is True
    var_26 = module_1.IPv6Address(var_10)
    var_27 = var_0.is_native_type(var_26)
    assert var_27 is True
    var_28 = var_0.is_native_type(var_1)
    assert var_28 is False
    var_29 = 123
    var_30 = var_0.is_native_type(var_29)
    assert var_30 is False
    var_31 = None
    var_32 = var_0.is_native_type(var_31)
    assert var_32 is False
    var_33 = var_0.serialize(var_31)
    assert var_33 is None
    var_34 = module_1.IPv4Address(var_1)
    var_35 = var_0.serialize(var_34)
    assert var_35 == '192.168.1.1'
    var_36 = module_1.IPv6Address(var_10)
    var_37 = var_0.serialize(var_36)
    assert var_37 == '::1'



# Parsed testcases at query #34
#--------------------------


import typesystem.formats as module_0
import ipaddress as module_1

def test_case_0():
    var_0 = module_0.IPAddressFormat()
    var_1 = '192.168.1.1'
    var_2 = '10.0.0.1'
    var_3 = '172.16.0.1'
    var_4 = '8.8.8.8'
    var_5 = '255.255.255.255'
    var_6 = '0.0.0.0'
    var_7 = '127.0.0.1'
    var_8 = [var_1, var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'
    var_10 = 'fe80::1'
    var_11 = '::1'
    var_12 = '2001:db8::1'
    var_13 = 'ff02::1'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'not_an_ip'
    var_16 = '192.168.1'
    var_17 = '192.168.1.256'
    var_18 = '192.168.1.1.1'
    var_19 = '2001:db8::1::'
    var_20 = '2001:db8:85a3:0000:0000:8a2e:0370:7334:extra'
    var_21 = [var_15, var_16, var_17, var_18, var_19, var_20]
    var_22 = '999.999.999.999'
    var_23 = '256.256.256.256'
    var_24 = '300.168.1.1'
    var_25 = [var_22, var_23, var_24]
    var_26 = module_1.IPv4Address(var_1)
    var_27 = var_0.is_native_type(var_26)
    assert var_27 is True
    var_28 = module_1.IPv6Address(var_11)
    var_29 = var_0.is_native_type(var_28)
    assert var_29 is True
    var_30 = var_0.is_native_type(var_1)
    assert var_30 is False
    var_31 = 123
    var_32 = var_0.is_native_type(var_31)
    assert var_32 is False
    var_33 = None
    var_34 = var_0.is_native_type(var_33)
    assert var_34 is False
    var_35 = module_1.IPv4Address(var_1)
    var_36 = module_1.IPv6Address(var_12)
    var_37 = var_0.serialize(var_35)
    assert var_37 == '192.168.1.1'
    var_38 = var_0.serialize(var_36)
    assert var_38 == '2001:db8::1'
    var_39 = var_0.serialize(var_33)
    assert var_39 is None
    var_40 = 'not_an_ip'
    var_41 = var_0.serialize(var_40)



