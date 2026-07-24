####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 5
    var_7 = 6
    var_8 = 10
    var_9 = 'x, y, id_'
    var_10 = 'Point2'
    var_11 = module_0.immutable(var_9, var_10)
    var_12 = 17
    var_13 = 18
    var_14 = 'a'
    var_15 = 'b'
    var_16 = [var_14, var_15]
    var_17 = 'Point3'
    var_18 = module_0.immutable(var_16, var_17)
    var_19 = 10
    var_20 = 20
    var_21 = 'p, q, r'
    var_22 = 'Point4'
    var_23 = module_0.immutable(var_21, var_22)
    var_24 = 'x,  y,  z'
    var_25 = 'Point5'
    var_26 = module_0.immutable(var_24, var_25)
    var_27 = ''
    var_28 = 'Empty'
    var_29 = module_0.immutable(var_27, var_28)
    var_30 = tuple()
    var_31 = -3
    var_32 = 'a, b_, c, d_'
    var_33 = 'Multi'
    var_34 = module_0.immutable(var_32, var_33)
    var_35 = 4
    var_36 = 30
    var_37 = 20
    var_38 = 40
    var_39 = 'Point6'
    var_40 = module_0.immutable(var_37, var_39)
    var_41 = 999
    var_42 = 'Point7'
    var_43 = module_0.immutable(var_41, var_42)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 5
    var_7 = 'x, y, id_'
    var_8 = 'PointFrozen'
    var_9 = module_0.immutable(var_7, var_8)
    var_10 = 17
    var_11 = 18
    var_12 = 'x'
    var_13 = 'y'
    var_14 = [var_12, var_13]
    var_15 = 'PointList'
    var_16 = module_0.immutable(var_14, var_15)
    var_17 = 5
    var_18 = 6
    var_19 = 'a, b, c'
    var_20 = 'PointComma'
    var_21 = module_0.immutable(var_19, var_20)
    var_22 = ''
    var_23 = 'Empty'
    var_24 = module_0.immutable(var_22, var_23)
    var_25 = 'Single'
    var_26 = module_0.immutable(var_12, var_25)
    var_27 = 42
    var_28 = 'x_, y_, z'
    var_29 = 'MultiFrozen'
    var_30 = module_0.immutable(var_28, var_29)
    var_31 = 10
    var_32 = 20
    var_33 = 30
    var_34 = tuple()
    var_35 = -3
    var_36 = 'a, b'
    var_37 = 'Verbose'
    var_38 = False
    var_39 = module_0.immutable(var_36, var_37, var_38)
    var_40 = 10
    var_41 = 20



# Parsed testcases at query #3
#--------------------------


import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 5
    var_7 = 6
    var_8 = 10
    var_9 = 'x, y, id_'
    var_10 = 'FrozenPoint'
    var_11 = module_0.immutable(var_9, var_10)
    var_12 = 17
    var_13 = 18
    var_14 = 5
    var_15 = 18
    var_16 = 'a, b, c'
    var_17 = 'Point2'
    var_18 = module_0.immutable(var_16, var_17)
    var_19 = 'x'
    var_20 = 'y'
    var_21 = [var_19, var_20]
    var_22 = 'Point3'
    var_23 = module_0.immutable(var_21, var_22)
    var_24 = 10
    var_25 = 20
    var_26 = ''
    var_27 = 'Empty'
    var_28 = module_0.immutable(var_26, var_27)
    var_29 = tuple()
    var_30 = -1
    var_31 = 2
    var_32 = -3
    var_33 = 'VerbosePoint'
    var_34 = False
    var_35 = module_0.immutable(var_32, var_33, var_34)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 5
    var_7 = 6
    var_8 = 10
    var_9 = 'x, y, id_'
    var_10 = 'Point_frozen'
    var_11 = module_0.immutable(var_9, var_10)
    var_12 = 17
    var_13 = 18
    var_14 = 'x'
    var_15 = 'y'
    var_16 = [var_14, var_15]
    var_17 = 'Point_list'
    var_18 = module_0.immutable(var_16, var_17)
    var_19 = 10
    var_20 = 20
    var_21 = 'x, y, z'
    var_22 = 'Point_spaces'
    var_23 = module_0.immutable(var_21, var_22)
    var_24 = ''
    var_25 = 'Empty'
    var_26 = module_0.immutable(var_24, var_25)
    var_27 = tuple()
    var_28 = -1
    var_29 = 2
    var_30 = -3
    var_31 = 'a, b_, c, d_'
    var_32 = 'MultiFreeze'
    var_33 = module_0.immutable(var_31, var_32)
    var_34 = 4
    var_35 = 30
    var_36 = 20
    var_37 = 40
    var_38 = 'Point_verbose'
    var_39 = True
    var_40 = module_0.immutable(var_37, var_38, var_39)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._immutable as module_0

def test_case_0():
    var_0 = 'x, y'
    var_1 = 'Point'
    var_2 = module_0.immutable(var_0, var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = [var_6, var_7, var_8]
    var_10 = 'Point2'
    var_11 = module_0.immutable(var_9, var_10)
    var_12 = 'x, y, id_'
    var_13 = 'FrozenPoint'
    var_14 = module_0.immutable(var_12, var_13)
    var_15 = 17
    var_16 = 5
    var_17 = 18
    var_18 = 10
    var_19 = 10
    var_20 = 20
    var_21 = ''
    var_22 = 'Empty'
    var_23 = module_0.immutable(var_21, var_22)
    var_24 = tuple()
    var_25 = -1
    var_26 = 2
    var_27 = -3
    var_28 = 'x, y, z'
    var_29 = 'Comma'
    var_30 = module_0.immutable(var_28, var_29)
    var_31 = 'VerbosePoint'
    var_32 = False
    var_33 = module_0.immutable(var_27, var_31, var_32)
    var_34 = 'a, b_, c, d_'
    var_35 = 'MultiFreeze'
    var_36 = module_0.immutable(var_34, var_35)
    var_37 = 4
    var_38 = 10
    var_39 = 30
    var_40 = 20
    var_41 = 40



