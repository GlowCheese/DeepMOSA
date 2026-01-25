####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = 2



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the method timestamp of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = 'INVALID_FORMAT'
    var_3 = var_1.timestamp(var_2)
    var_4 = 2000
    var_5 = 2001
    var_6 = var_1.timestamp()



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 10
    var_4 = 'days'
    var_5 = {var_4: var_2}
    var_6 = 2
    var_7 = 11



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test method bulk_create_datetimes of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = 2023
    var_3 = 1
    var_4 = 5
    var_5 = 2
    var_6 = 0
    var_7 = 23
    var_8 = 59
    var_9 = 12
    var_10 = 30
    var_11 = 0
    var_12 = None



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'Unit test for method bulk_create_datetimes of class Datetime.'
    var_1 = 2023
    var_2 = 1
    var_3 = 10
    var_4 = 2
    var_5 = 1



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Unit test for method duration of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = 1
    var_3 = 10
    var_4 = None
    var_5 = var_1.duration(var_2, var_3, var_4)
    var_6 = 10
    var_7 = 1
    var_8 = var_1.duration(var_6, var_7, var_3)
    var_9 = '1'
    var_10 = '10'
    var_11 = var_1.duration(var_9, var_10, var_3)



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 10
    var_4 = 2
    var_5 = 0
    var_6 = 1
    var_7 = 0



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 2020
    var_1 = 1
    var_2 = 10
    var_3 = module_0.Datetime()
    var_4 = 2
    var_5 = module_0.Datetime()
    var_6 = module_0.Datetime()
    var_7 = 1
    var_8 = module_0.Datetime()
    var_9 = -1
    var_10 = module_0.Datetime()
    var_11 = 0
    var_12 = module_0.Datetime()



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Unit test for method duration of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = None
    var_6 = var_1.duration(duration_unit=var_5)
    var_7 = 10
    var_8 = 5
    var_9 = var_1.duration(var_7, var_8)
    var_10 = 'invalid'
    var_11 = 10
    var_12 = var_1.duration(var_10, var_11)
    var_13 = 5
    var_14 = 'invalid'
    var_15 = var_1.duration(var_13, var_14)



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the duration method of the Datetime class.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = None
    var_6 = var_1.duration(duration_unit=var_5)
    var_7 = 10
    var_8 = 5
    var_9 = var_1.duration(var_7, var_8)
    var_10 = 1.5
    var_11 = 2.5
    var_12 = var_1.duration(var_10, var_11)



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = None
    var_4 = var_0.duration(var_1, var_2, var_3)



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test method datetime of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = var_1.datetime()
    var_3 = 2010
    var_4 = 2020
    var_5 = var_1.datetime(var_3, var_4)
    var_6 = 'UTC'
    var_7 = var_1.datetime(timezone=var_6)



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2024
    var_2 = var_0.datetime(var_1, var_1)



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 2



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the duration method.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = var_1.duration(var_3, var_4)
    var_6 = 10
    var_7 = 5
    var_8 = var_1.duration(var_6, var_7)
    var_9 = 'invalid'
    var_10 = 10
    var_11 = var_1.duration(var_9, var_10)
    var_12 = 1
    var_13 = 'invalid'
    var_14 = var_1.duration(var_12, var_13)
    var_15 = None
    var_16 = var_1.duration(duration_unit=var_15)



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 60
    var_3 = 1
    var_4 = 10
    var_5 = 5
    var_6 = 15
    var_7 = var_0.duration(var_5, var_6)
    var_8 = 3600
    var_9 = 1000
    var_10 = 1000000.0
    var_11 = 604800
    var_12 = 86400
    var_13 = 10
    var_14 = 5
    var_15 = var_0.duration(var_13, var_14)
    var_16 = '1'
    var_17 = 10
    var_18 = var_0.duration(var_16, var_17)
    var_19 = 1
    var_20 = '10'
    var_21 = var_0.duration(var_19, var_20)
    var_22 = None
    var_23 = var_0.duration(duration_unit=var_22)
    var_24 = 'invalid'
    var_25 = var_0.duration(duration_unit=var_24)



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = var_0.duration(var_2, var_3)
    var_5 = 10
    var_6 = 5
    var_7 = var_0.duration(var_5, var_6)
    var_8 = 1.5
    var_9 = 10
    var_10 = var_0.duration(var_8, var_9)
    var_11 = 1
    var_12 = 10.5
    var_13 = var_0.duration(var_11, var_12)



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2021
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 0
    var_6 = 2000
    var_7 = 1000
    var_8 = 0
    var_9 = -1
    var_10 = -1000
    var_11 = 0
    var_12 = 0



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the duration method of the Datetime class.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = None
    var_6 = var_1.duration(duration_unit=var_5)
    var_7 = 10
    var_8 = 5
    var_9 = var_1.duration(var_7, var_8)
    var_10 = '1'
    var_11 = 5
    var_12 = var_1.duration(var_10, var_11)



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = None
    var_5 = var_0.duration(duration_unit=var_4)
    var_6 = 20
    var_7 = 10
    var_8 = var_0.duration(var_6, var_7)
    var_9 = 1.5
    var_10 = 10.5
    var_11 = var_0.duration(var_9, var_10)



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2023
    var_3 = var_0.datetime(var_1, var_2)



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = 10
    var_4 = 1
    var_5 = 1.5
    var_6 = 10
    var_7 = 1
    var_8 = 10.5



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = var_1.hour
    var_3 = var_1.minute
    var_4 = var_1.second
    var_5 = var_1.microsecond



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = None
    var_5 = var_0.duration(duration_unit=var_4)
    var_6 = 10
    var_7 = 5
    var_8 = var_0.duration(var_6, var_7)
    var_9 = 1.5
    var_10 = 5
    var_11 = var_0.duration(var_9, var_10)
    var_12 = 1
    var_13 = 5.5
    var_14 = var_0.duration(var_12, var_13)



# Parsed testcases at query #26
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test method duration of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = var_1.duration(var_3, var_4)
    var_6 = None
    var_7 = var_1.duration(duration_unit=var_6)
    var_8 = 10
    var_9 = 5
    var_10 = var_1.duration(var_8, var_9)
    var_11 = 1.5
    var_12 = 5.5
    var_13 = var_1.duration(var_11, var_12)



# Parsed testcases at query #27
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10
    var_3 = None
    var_4 = var_0.duration(var_1, var_2, var_3)
    var_5 = 10
    var_6 = 1
    var_7 = 1.5
    var_8 = 10.5



# Parsed testcases at query #28
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = var_0.duration(var_2, var_3)
    var_5 = 2
    var_6 = 8
    var_7 = None
    var_8 = var_0.duration(duration_unit=var_7)
    var_9 = 10
    var_10 = 5
    var_11 = var_0.duration(var_9, var_10)
    var_12 = 1.5
    var_13 = 5.5
    var_14 = var_0.duration(var_12, var_13)
    var_15 = 'invalid'
    var_16 = var_0.duration(duration_unit=var_15)



# Parsed testcases at query #29
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 1
    var_2 = 10



# Parsed testcases at query #30
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the method bulk_create_datetimes of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = 2020
    var_3 = 1
    var_4 = 10
    var_5 = 2
    var_6 = 0
    var_7 = 5
    var_8 = 1
    var_9 = -1



# Parsed testcases at query #31
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2022
    var_2 = 2023
    var_3 = 'UTC'
    var_4 = var_0.datetime(var_1, var_2, var_3)



# Parsed testcases at query #32
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Unit test for method datetime of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = var_1.datetime()



# Parsed testcases at query #33
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = var_0.duration(var_2, var_3)
    var_5 = 10
    var_6 = 5
    var_7 = var_0.duration(var_5, var_6)
    var_8 = 'invalid'
    var_9 = 5
    var_10 = var_0.duration(var_8, var_9)
    var_11 = 5
    var_12 = 'invalid'
    var_13 = var_0.duration(var_11, var_12)



# Parsed testcases at query #34
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2020
    var_3 = 2021
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = 'America/New_York'
    var_8 = var_0.datetime(timezone=var_7)



# Parsed testcases at query #35
#--------------------------


import mimesis.providers.date as module_0
import datetime as module_1

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 5
    var_2 = module_1.timedelta()
    var_3 = 1
    var_4 = module_1.timedelta()
    var_5 = module_1.timedelta()
    var_6 = 1
    var_7 = module_1.timedelta()
    var_8 = None
    var_9 = 1



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test for method bulk_create_datetimes of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = 2023
    var_3 = 1
    var_4 = 0
    var_5 = 10
    var_6 = 'days'
    var_7 = {var_6: var_3}
    var_8 = 2
    var_9 = 11



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 0
    var_6 = 4
    var_7 = 4000
    var_8 = 1000
    var_9 = 1
    var_10 = 0



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the duration method of the Datetime class.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = None
    var_6 = var_1.duration(duration_unit=var_5)
    var_7 = 10
    var_8 = 5
    var_9 = var_1.duration(var_7, var_8)
    var_10 = '1'
    var_11 = 5
    var_12 = var_1.duration(var_10, var_11)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the duration method of the Datetime class.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = var_1.duration(var_3, var_4)
    var_6 = None
    var_7 = var_1.duration(duration_unit=var_6)
    var_8 = 10
    var_9 = 5
    var_10 = var_1.duration(var_8, var_9)
    var_11 = 1.5
    var_12 = 10
    var_13 = var_1.duration(var_11, var_12)
    var_14 = 1
    var_15 = 10.5
    var_16 = var_1.duration(var_14, var_15)



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test method bulk_create_datetimes of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = 2020
    var_3 = 1
    var_4 = 10
    var_5 = 2
    var_6 = 12
    var_7 = 0
    var_8 = 1
    var_9 = -1



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2000
    var_3 = 2020
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = 'Invalid/Timezone'
    var_8 = var_0.datetime(timezone=var_7)



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the datetime method of the Datetime class.'
    var_1 = module_0.Datetime()
    var_2 = var_1.datetime()
    var_3 = 2000
    var_4 = 2010
    var_5 = var_1.datetime(var_3, var_4)
    var_6 = 'UTC'
    var_7 = var_1.datetime(timezone=var_6)
    var_8 = 'Invalid/Timezone'
    var_9 = var_1.datetime(timezone=var_8)



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test bulk_create_datetimes method of Datetime class.'
    var_1 = module_0.Datetime()
    var_2 = 2020
    var_3 = 1
    var_4 = 10
    var_5 = 2
    var_6 = 12
    var_7 = 0
    var_8 = 1
    var_9 = 0



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = var_0.duration(var_2, var_3)
    var_5 = 2
    var_6 = var_0.duration(var_5, var_5)
    var_7 = 10
    var_8 = 5
    var_9 = var_0.duration(var_7, var_8)
    var_10 = '1'
    var_11 = 5
    var_12 = var_0.duration(var_10, var_11)
    var_13 = 1
    var_14 = '5'
    var_15 = var_0.duration(var_13, var_14)



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Unit test for method datetime of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = var_1.datetime()
    var_3 = 2000
    var_4 = 2010
    var_5 = var_1.datetime(var_3, var_4)
    var_6 = 'UTC'
    var_7 = var_1.datetime(timezone=var_6)



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the datetime method of the Datetime class.'
    var_1 = module_0.Datetime()
    var_2 = var_1.datetime()
    var_3 = 2000
    var_4 = 2010
    var_5 = var_1.datetime(var_3, var_4)
    var_6 = 'UTC'
    var_7 = var_1.datetime(timezone=var_6)
    var_8 = 'Invalid/Timezone'
    var_9 = var_1.datetime(timezone=var_8)



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Unit test for method duration of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = var_1.duration(var_3, var_4)
    var_6 = 2
    var_7 = var_1.duration(var_6, var_6)
    var_8 = 10
    var_9 = 5
    var_10 = var_1.duration(var_8, var_9)
    var_11 = '1'
    var_12 = 5
    var_13 = var_1.duration(var_11, var_12)
    var_14 = 1
    var_15 = '5'
    var_16 = var_1.duration(var_14, var_15)
    var_17 = None
    var_18 = var_1.duration(duration_unit=var_17)



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 10
    var_4 = 2
    var_5 = 11



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Unit test for method datetime of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = 2020
    var_3 = 2023
    var_4 = var_1.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_1.datetime(timezone=var_5)
    var_7 = 'Invalid/Timezone'
    var_8 = var_1.datetime(timezone=var_7)



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 2022
    var_4 = 12
    var_5 = 31
    var_6 = 1
    var_7 = 2
    var_8 = 0
    var_9 = 3
    var_10 = 0



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = var_0.duration(var_2, var_3)
    var_5 = None
    var_6 = var_0.duration(duration_unit=var_5)
    var_7 = 10
    var_8 = 5
    var_9 = var_0.duration(var_7, var_8)
    var_10 = '1'
    var_11 = 5
    var_12 = var_0.duration(var_10, var_11)
    var_13 = 1
    var_14 = '5'
    var_15 = var_0.duration(var_13, var_14)



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = var_0.duration(var_2, var_3)
    var_5 = 10
    var_6 = 5
    var_7 = var_0.duration(var_5, var_6)
    var_8 = 'invalid'
    var_9 = 10
    var_10 = var_0.duration(var_8, var_9)
    var_11 = 5
    var_12 = 'invalid'
    var_13 = var_0.duration(var_11, var_12)



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 10
    var_4 = 2



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test method datetime of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = var_1.datetime()
    var_3 = 2000
    var_4 = 2010
    var_5 = var_1.datetime(var_3, var_4)
    var_6 = 'UTC'
    var_7 = var_1.datetime(timezone=var_6)
    var_8 = 'Invalid/Timezone'
    var_9 = var_1.datetime(timezone=var_8)



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 10
    var_4 = 2
    var_5 = None
    var_6 = 1
    var_7 = 1
    var_8 = 0
    var_9 = 12
    var_10 = 0



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Unit test for method duration of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = var_1.duration(var_3, var_4)
    var_6 = 2
    var_7 = var_1.duration(var_6, var_6)
    var_8 = 10
    var_9 = 5
    var_10 = var_1.duration(var_8, var_9)
    var_11 = '1'
    var_12 = 5
    var_13 = var_1.duration(var_11, var_12)
    var_14 = 1
    var_15 = '5'
    var_16 = var_1.duration(var_14, var_15)



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the duration method of the Datetime class.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = var_1.duration(var_3, var_4)
    var_6 = None
    var_7 = var_1.duration(duration_unit=var_6)
    var_8 = 10
    var_9 = 5
    var_10 = var_1.duration(var_8, var_9)
    var_11 = 'a'
    var_12 = 5
    var_13 = var_1.duration(var_11, var_12)



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 2



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test bulk_create_datetimes method of Datetime class.'
    var_1 = module_0.Datetime()
    var_2 = 2020
    var_3 = 1
    var_4 = 10
    var_5 = 'days'
    var_6 = {var_5: var_3}
    var_7 = 2



# Parsed testcases at query #26
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the method bulk_create_datetimes of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = 2023
    var_3 = 1
    var_4 = 10
    var_5 = 2
    var_6 = 12
    var_7 = 0
    var_8 = 30
    var_9 = 1
    var_10 = -1



# Parsed testcases at query #27
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the method duration of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = None
    var_6 = var_1.duration(duration_unit=var_5)
    var_7 = 10
    var_8 = 5
    var_9 = var_1.duration(var_7, var_8)
    var_10 = '1'
    var_11 = 5
    var_12 = var_1.duration(var_10, var_11)



# Parsed testcases at query #28
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 2
    var_3 = 5
    var_4 = 1
    var_5 = 3
    var_6 = None
    var_7 = var_0.duration(var_4, var_5, var_6)
    var_8 = 5
    var_9 = 1
    var_10 = var_0.duration(var_8, var_9)
    var_11 = 1.5
    var_12 = 3.5
    var_13 = var_0.duration(var_11, var_12)



# Parsed testcases at query #29
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Unit test for method duration of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = var_1.duration()
    var_3 = 5
    var_4 = 15
    var_5 = var_1.duration(var_3, var_4)
    var_6 = None
    var_7 = var_1.duration(duration_unit=var_6)
    var_8 = 10
    var_9 = 5
    var_10 = var_1.duration(var_8, var_9)
    var_11 = '1'
    var_12 = 5
    var_13 = var_1.duration(var_11, var_12)
    var_14 = 1
    var_15 = '5'
    var_16 = var_1.duration(var_14, var_15)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'Test the method bulk_create_datetimes of class Datetime.'
    var_1 = 2020
    var_2 = 1
    var_3 = 10
    var_4 = 'days'
    var_5 = {var_4: var_2}
    var_6 = 2
    var_7 = 'hours'
    var_8 = 12
    var_9 = {var_7: var_8}
    var_10 = 0



# Parsed testcases at query #31
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test the datetime method of the Datetime class.'
    var_1 = module_0.Datetime()
    var_2 = var_1.datetime()
    var_3 = 2000
    var_4 = 2010
    var_5 = var_1.datetime(var_3, var_4)
    var_6 = 'UTC'
    var_7 = var_1.datetime(timezone=var_6)



# Parsed testcases at query #32
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test method bulk_create_datetimes of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = 2020
    var_3 = 1
    var_4 = 10
    var_5 = 2
    var_6 = 30
    var_7 = 0
    var_8 = 1
    var_9 = 0



# Parsed testcases at query #33
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = 'Test for method bulk_create_datetimes of class Datetime.'
    var_1 = module_0.Datetime()
    var_2 = 2020
    var_3 = 1
    var_4 = 10
    var_5 = 2



# Parsed testcases at query #34
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2010
    var_3 = 2020
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'Europe/London'
    var_6 = var_0.datetime(timezone=var_5)



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 2023
    var_1 = 1
    var_2 = 3
    var_3 = 2
    var_4 = 1
    var_5 = 0



