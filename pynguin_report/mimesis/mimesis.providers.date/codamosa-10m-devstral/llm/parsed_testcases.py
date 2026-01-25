####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2000
    var_2 = 2023
    var_3 = 'UTC'
    var_4 = var_0.datetime(var_1, var_2)
    var_5 = var_0.datetime(var_1, var_2, var_3)
    var_6 = var_5.tzinfo
    var_7 = str(var_6)
    var_8 = var_0.datetime()



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 4
    var_6 = 0
    var_7 = 2023
    var_8 = 1
    var_9 = 3
    var_10 = 2023
    var_11 = 1
    var_12 = 3
    var_13 = 2023
    var_14 = 1
    var_15 = 3
    var_16 = 0
    var_17 = None
    var_18 = 1



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2023
    var_3 = var_0.datetime(var_1, var_2)
    var_4 = 'UTC'
    var_5 = var_0.datetime(var_1, var_2, var_4)
    var_6 = var_5.tzinfo
    var_7 = str(var_6)
    var_8 = var_0.datetime()
    var_9 = 'Invalid/Timezone'
    var_10 = var_0.datetime(var_1, var_2, var_9)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 60
    var_3 = 3600
    var_4 = 5
    var_5 = 15
    var_6 = None
    var_7 = var_0.duration(duration_unit=var_6)
    var_8 = 10
    var_9 = 1
    var_10 = var_0.duration(var_8, var_9)
    var_11 = '1'
    var_12 = 10
    var_13 = var_0.duration(var_11, var_12)
    var_14 = 1
    var_15 = '10'
    var_16 = var_0.duration(var_14, var_15)



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2000
    var_3 = 2010
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = var_6.tzinfo
    var_8 = str(var_7)
    assert var_8 == 'UTC'
    var_9 = 'Invalid/Timezone'
    var_10 = var_0.datetime(timezone=var_9)
    var_11 = 2020
    var_12 = var_0.datetime(var_11, var_11)



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 60
    var_3 = 3600
    var_4 = 5
    var_5 = 15
    var_6 = var_0.duration(var_4, var_5)
    var_7 = None
    var_8 = var_0.duration(duration_unit=var_7)
    var_9 = 10
    var_10 = 1
    var_11 = var_0.duration(var_9, var_10)
    var_12 = '1'
    var_13 = 10
    var_14 = var_0.duration(var_12, var_13)
    var_15 = 1
    var_16 = '10'
    var_17 = var_0.duration(var_15, var_16)



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 12
    var_6 = 0
    var_7 = None
    var_8 = 1
    var_9 = 1
    var_10 = 0



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 12
    var_6 = 0
    var_7 = 30
    var_8 = 15
    var_9 = 500000
    var_10 = None
    var_11 = 0



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2000
    var_3 = 2010
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = var_6.tzinfo
    var_8 = str(var_7)
    assert var_8 == 'UTC'
    var_9 = 'UTC'
    var_10 = var_0.datetime(timezone=var_9)



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2020
    var_3 = 2025
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = var_6.tzinfo
    var_8 = str(var_7)
    assert var_8 == 'UTC'
    var_9 = 'UTC'
    var_10 = var_0.datetime(timezone=var_9)
    var_11 = None



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 60
    var_3 = 5
    var_4 = 15
    var_5 = 1
    var_6 = 24
    var_7 = 3600
    var_8 = 30
    var_9 = 4
    var_10 = 10
    var_11 = 1
    var_12 = var_0.duration(var_10, var_11)
    var_13 = '1'
    var_14 = 10
    var_15 = var_0.duration(var_13, var_14)
    var_16 = 1
    var_17 = '10'
    var_18 = var_0.duration(var_16, var_17)



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 0
    var_6 = None
    var_7 = 1
    var_8 = None
    var_9 = 1
    var_10 = 1
    var_11 = 0
    var_12 = -1



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 10
    var_4 = 2
    var_5 = 0
    var_6 = 5
    var_7 = None
    var_8 = 1
    var_9 = 1
    var_10 = 0
    var_11 = -1



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2020
    var_3 = 2025
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = var_6.tzinfo
    var_8 = str(var_7)
    assert var_8 == 'UTC'
    var_9 = 'UTC'
    var_10 = var_0.datetime(timezone=var_9)
    var_11 = None



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 60
    var_3 = 3600
    var_4 = 5
    var_5 = 15
    var_6 = var_0.duration(var_4, var_5)
    var_7 = None
    var_8 = var_0.duration(duration_unit=var_7)
    var_9 = 10
    var_10 = 1
    var_11 = var_0.duration(var_9, var_10)
    var_12 = '1'
    var_13 = 10
    var_14 = var_0.duration(var_12, var_13)
    var_15 = 1
    var_16 = '10'
    var_17 = var_0.duration(var_15, var_16)



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 0
    var_6 = None
    var_7 = 1
    var_8 = None
    var_9 = 1
    var_10 = 1
    var_11 = 0
    var_12 = -1



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 60
    var_3 = 3600
    var_4 = 5
    var_5 = 15
    var_6 = var_0.duration(var_4, var_5)
    var_7 = None
    var_8 = var_0.duration(duration_unit=var_7)
    var_9 = 10
    var_10 = 1
    var_11 = var_0.duration(var_9, var_10)
    var_12 = '1'
    var_13 = 10
    var_14 = var_0.duration(var_12, var_13)
    var_15 = 1
    var_16 = '10'
    var_17 = var_0.duration(var_15, var_16)



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = 1
    var_5 = 10
    var_6 = None
    var_7 = var_0.duration(var_4, var_5, var_6)
    var_8 = 10
    var_9 = 1
    var_10 = var_0.duration(var_8, var_9)
    var_11 = '1'
    var_12 = 10
    var_13 = var_0.duration(var_11, var_12)
    var_14 = 1
    var_15 = '10'
    var_16 = var_0.duration(var_14, var_15)



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 0
    var_6 = 3
    var_7 = 10
    var_8 = 30
    var_9 = None
    var_10 = 1
    var_11 = 2023
    var_12 = 1
    var_13 = 5
    var_14 = 2023
    var_15 = 1
    var_16 = 5
    var_17 = 0
    var_18 = 2023
    var_19 = 1
    var_20 = 5
    var_21 = -1



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 60
    var_3 = 5
    var_4 = 15
    var_5 = var_0.duration(var_3, var_4)
    var_6 = 3600
    var_7 = None
    var_8 = var_0.duration(duration_unit=var_7)
    var_9 = 10
    var_10 = 1
    var_11 = var_0.duration(var_9, var_10)
    var_12 = '1'
    var_13 = 10
    var_14 = var_0.duration(var_12, var_13)
    var_15 = 1
    var_16 = '10'
    var_17 = var_0.duration(var_15, var_16)



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 0
    var_6 = 2000
    var_7 = 1000
    var_8 = None
    var_9 = 1
    var_10 = 2020
    var_11 = 1
    var_12 = 3
    var_13 = 2020
    var_14 = 1
    var_15 = 3
    var_16 = 0



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 0
    var_6 = 1
    var_7 = 0
    var_8 = -1



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 0
    var_6 = 5000
    var_7 = 1000
    var_8 = None
    var_9 = 1
    var_10 = None
    var_11 = 1
    var_12 = 1
    var_13 = 0
    var_14 = -1



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 0
    var_6 = 12
    var_7 = 1
    var_8 = 0
    var_9 = None
    var_10 = 1



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = None
    var_6 = 0



# Parsed testcases at query #26
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
    var_7 = var_6.tzinfo
    var_8 = str(var_7)
    assert var_8 == 'UTC'
    var_9 = 'Invalid/Timezone'
    var_10 = var_0.datetime(timezone=var_9)
    var_11 = var_0.datetime()
    var_12 = var_0.datetime()



# Parsed testcases at query #27
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = 1
    var_5 = 30
    var_6 = 10
    var_7 = 1
    var_8 = var_0.duration(var_6, var_7)
    var_9 = '1'
    var_10 = 10
    var_11 = var_0.duration(var_9, var_10)



# Parsed testcases at query #28
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 0
    var_6 = 5000
    var_7 = 1000
    var_8 = 30
    var_9 = None
    var_10 = 1
    var_11 = 2020
    var_12 = 1
    var_13 = 5
    var_14 = 2020
    var_15 = 1
    var_16 = 5
    var_17 = 0
    var_18 = 2020
    var_19 = 1
    var_20 = 5
    var_21 = -1



# Parsed testcases at query #29
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
    var_8 = 1
    var_9 = var_0.duration(var_7, var_8)
    var_10 = 'invalid'
    var_11 = 10
    var_12 = var_0.duration(var_10, var_11)
    var_13 = 1
    var_14 = 'invalid'
    var_15 = var_0.duration(var_13, var_14)



# Parsed testcases at query #30
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2020
    var_3 = 2025
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = var_6.tzinfo
    var_8 = str(var_7)
    assert var_8 == 'UTC'
    var_9 = 'Invalid/Timezone'
    var_10 = var_0.datetime(timezone=var_9)
    var_11 = 2025
    var_12 = 2020
    var_13 = var_0.datetime(var_11, var_12)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2020
    var_3 = 2022
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = 'Invalid/Timezone'
    var_8 = var_0.datetime(timezone=var_7)
    var_9 = 2022
    var_10 = 2020
    var_11 = var_0.datetime(var_9, var_10)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 4
    var_6 = None
    var_7 = 2020
    var_8 = 1
    var_9 = 3
    var_10 = 0
    var_11 = -1



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 4
    var_6 = 0
    var_7 = 2020
    var_8 = 1
    var_9 = 3
    var_10 = 2020
    var_11 = 1
    var_12 = 3
    var_13 = 0
    var_14 = 2020
    var_15 = 1
    var_16 = 3
    var_17 = None
    var_18 = 1



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 0
    var_6 = 3
    var_7 = 5000
    var_8 = 1000
    var_9 = 1
    var_10 = 0



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 0
    var_6 = 2000
    var_7 = 1000
    var_8 = None
    var_9 = 1
    var_10 = 2020
    var_11 = 1
    var_12 = 3
    var_13 = 2020
    var_14 = 1
    var_15 = 3
    var_16 = 0
    var_17 = 2020
    var_18 = 1
    var_19 = 3
    var_20 = -1



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 0
    var_6 = 3
    var_7 = 10
    var_8 = 30
    var_9 = 8
    var_10 = None
    var_11 = 1
    var_12 = 1
    var_13 = 0
    var_14 = -1



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 4
    var_6 = 0
    var_7 = 1
    var_8 = 0



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 4
    var_6 = 0
    var_7 = None
    var_8 = 1
    var_9 = None
    var_10 = 1
    var_11 = 1
    var_12 = 0
    var_13 = 30



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 'Z'
    var_2 = 2020
    var_3 = 2021
    var_4 = var_0.timestamp()



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 10
    var_4 = 2
    var_5 = 6
    var_6 = 0
    var_7 = 30
    var_8 = 1
    var_9 = 0
    var_10 = -1



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 12
    var_6 = 0
    var_7 = 30
    var_8 = 1
    var_9 = 0
    var_10 = None
    var_11 = 1



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 4
    var_6 = 0
    var_7 = None
    var_8 = 1
    var_9 = None
    var_10 = 1
    var_11 = 1
    var_12 = 0
    var_13 = -1



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 0
    var_6 = None
    var_7 = 1
    var_8 = None
    var_9 = 1
    var_10 = 1
    var_11 = 0
    var_12 = -1



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2020
    var_3 = 2022
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = var_6.tzinfo
    var_8 = str(var_7)
    assert var_8 == 'UTC'
    var_9 = 'UTC'
    var_10 = var_0.datetime(timezone=var_9)
    var_11 = None



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 60
    var_3 = 3600
    var_4 = 5
    var_5 = 15
    var_6 = var_0.duration(var_4, var_5)
    var_7 = None
    var_8 = var_0.duration(duration_unit=var_7)
    var_9 = 10
    var_10 = 1
    var_11 = var_0.duration(var_9, var_10)
    var_12 = '1'
    var_13 = 10
    var_14 = var_0.duration(var_12, var_13)



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 3
    var_4 = 2
    var_5 = 0
    var_6 = 1
    var_7 = 0



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2023
    var_3 = 'UTC'
    var_4 = var_0.datetime(var_1, var_2)
    var_5 = var_0.datetime(var_1, var_2, var_3)
    var_6 = var_0.datetime()



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 5
    var_3 = 15
    var_4 = 1
    var_5 = 10
    var_6 = None
    var_7 = var_0.duration(var_4, var_5, var_6)
    var_8 = 10
    var_9 = 1
    var_10 = var_0.duration(var_8, var_9)
    var_11 = '1'
    var_12 = 10
    var_13 = var_0.duration(var_11, var_12)
    var_14 = 1
    var_15 = '10'
    var_16 = var_0.duration(var_14, var_15)



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2023
    var_3 = 'UTC'
    var_4 = var_0.datetime(var_1, var_2)
    var_5 = var_0.datetime(var_1, var_2, var_3)
    var_6 = var_5.tzinfo
    var_7 = str(var_6)
    var_8 = var_0.datetime()



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 60
    var_3 = 5
    var_4 = 15
    var_5 = var_0.duration(var_3, var_4)
    var_6 = 3600
    var_7 = None
    var_8 = var_0.duration(duration_unit=var_7)
    var_9 = 1
    var_10 = var_0.duration(var_9, var_9)
    var_11 = 10
    var_12 = 1
    var_13 = var_0.duration(var_11, var_12)
    var_14 = '1'
    var_15 = 10
    var_16 = var_0.duration(var_14, var_15)
    var_17 = 1
    var_18 = '10'
    var_19 = var_0.duration(var_17, var_18)



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.duration()
    var_2 = 60
    var_3 = 3600
    var_4 = 5
    var_5 = 15
    var_6 = var_0.duration(var_4, var_5)
    var_7 = None
    var_8 = var_0.duration(duration_unit=var_7)
    var_9 = 10
    var_10 = 1
    var_11 = var_0.duration(var_9, var_10)
    var_12 = '1'
    var_13 = 10
    var_14 = var_0.duration(var_12, var_13)
    var_15 = 1
    var_16 = '10'
    var_17 = var_0.duration(var_15, var_16)



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2023
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 0
    var_6 = 3
    var_7 = 10
    var_8 = 30
    var_9 = 8
    var_10 = None
    var_11 = 1
    var_12 = None
    var_13 = 1
    var_14 = 1
    var_15 = 0
    var_16 = -1



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = var_0.datetime()
    var_2 = 2010
    var_3 = 2020
    var_4 = var_0.datetime(var_2, var_3)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = var_6.tzinfo
    var_8 = str(var_7)
    assert var_8 == 'UTC'
    var_9 = 'Invalid/Timezone'
    var_10 = var_0.datetime(timezone=var_9)



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 2023
    var_3 = var_0.datetime()
    var_4 = var_0.datetime(var_1, var_2)
    var_5 = 'UTC'
    var_6 = var_0.datetime(timezone=var_5)
    var_7 = var_6.tzinfo
    var_8 = str(var_7)
    var_9 = 'Invalid/Timezone'
    var_10 = var_0.datetime(timezone=var_9)



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.date as module_0

def test_case_0():
    var_0 = module_0.Datetime()
    var_1 = 2020
    var_2 = 1
    var_3 = 5
    var_4 = 2
    var_5 = 12
    var_6 = 0
    var_7 = 30
    var_8 = 15
    var_9 = 500000
    var_10 = 1
    var_11 = 0
    var_12 = None
    var_13 = 1



