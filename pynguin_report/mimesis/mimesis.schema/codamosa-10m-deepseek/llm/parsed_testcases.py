####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 'All tests passed!'
    var_2 = print(var_1)



# Parsed testcases at query #2
#--------------------------


import mimesis.schema as module_0


def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = 'test_schema'
    var_2 = 'id'
    var_3 = 'name'
    var_4 = 1
    var_5 = 'test'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = lambda : var_6
    var_8 = 2
    var_9 = 0
    var_10 = module_0.SchemaContext(var_9, builder=var_0)
    var_11 = var_10.ref(var_1)
    var_12 = module_0.SchemaContext(var_9)
    var_13 = 'test_schema'
    var_14 = var_12.ref(var_13)
    var_15 = module_0.SchemaContext(var_9, builder=var_0)
    var_16 = 'non_existent_schema'
    var_17 = var_15.ref(var_16)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'handler1'
    var_2 = lambda random, **kwargs: var_1
    var_3 = 'handler2'
    var_4 = lambda random, **kwargs: var_3
    var_5 = 'field1'
    var_6 = (var_5, var_2)
    var_7 = 'field2'
    var_8 = (var_7, var_4)
    var_9 = [var_6, var_8]
    var_10 = var_0.register_handlers(var_9)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 2



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = 'test_schema'
    var_2 = 'id'
    var_3 = 'name'
    var_4 = 1
    var_5 = 'test'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = lambda : var_6
    var_8 = 0
    var_9 = module_0.SchemaContext(var_8, builder=var_0)
    var_10 = var_9.pick_from(var_1, var_3)
    assert var_10 == 'test'
    var_11 = var_9.pick_from(var_1)
    var_12 = 'non_existent_schema'
    var_13 = var_9.pick_from(var_12)
    var_14 = module_0.SchemaContext(var_8)
    var_15 = 'test_schema'
    var_16 = var_14.pick_from(var_15)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Fieldset()
    var_1 = 'username'
    var_2 = 5
    var_3 = module_0.Fieldset()
    var_4 = module_0.Fieldset()
    var_5 = 3
    var_6 = module_0.Fieldset()
    var_7 = 'username'
    var_8 = 0
    var_9 = module_0.Fieldset()
    var_10 = 'custom_field'
    var_11 = 1
    var_12 = 100
    var_13 = lambda r, **kwargs: r.randint(var_11, var_12)
    var_14 = 4
    var_15 = module_0.Fieldset()
    var_16 = 'alias'
    var_17 = 2
    var_18 = module_0.Fieldset()
    var_19 = lambda x: x.upper()
    var_20 = module_0.Fieldset()
    var_21 = 10
    var_22 = lambda x, r: x + str(r.randint(var_11, var_21))
    var_23 = module_0.Fieldset()
    var_24 = 'person.full_name'
    var_25 = module_0.Fieldset()
    var_26 = 'full_name'
    var_27 = module_0.Fieldset()
    var_28 = 'invalid_field'
    var_29 = 2
    var_30 = module_0.Fieldset()
    var_31 = 'person:full_name'
    var_32 = module_0.Fieldset()
    var_33 = 'person full_name'
    var_34 = module_0.Fieldset()
    var_35 = 'person/full_name'
    var_36 = module_0.Fieldset()
    var_37 = 'person.full.name'
    var_38 = 2
    var_39 = module_0.Fieldset()
    var_40 = 'custom'
    var_41 = lambda r, prefix='', **kwargs: prefix + str(r.randint(var_11, var_12))
    var_42 = 'num_'
    var_43 = module_0.Fieldset()
    var_44 = 'custom_value'
    var_45 = lambda r, **kwargs: var_44
    var_46 = 'custom'
    var_47 = 2
    var_48 = module_0.Fieldset()
    var_49 = 'custom1'
    var_50 = 'value1'
    var_51 = lambda r, **kwargs: var_50
    var_52 = 'custom2'
    var_53 = 'value2'
    var_54 = lambda r, **kwargs: var_53
    var_55 = 'custom1'
    var_56 = 2
    var_57 = 'custom2'
    var_58 = 2
    var_59 = 42
    var_60 = module_0.Fieldset()
    var_61 = module_0.Fieldset()
    var_62 = module_0.Fieldset()
    var_63 = module_0.Fieldset()
    var_64 = 123
    var_65 = 'alias'
    var_66 = 2
    var_67 = module_0.Fieldset()
    var_68 = module_0.Fieldset()
    var_69 = module_0.Fieldset()
    var_70 = 'alias'
    var_71 = 2
    var_72 = module_0.Fieldset()
    var_73 = 'alias'
    var_74 = 2
    var_75 = module_0.Fieldset()
    var_76 = 'alias1'
    var_77 = 'alias2'
    var_78 = 'alias1'
    var_79 = 2
    var_80 = module_0.Fieldset()
    var_81 = 'alias'
    var_82 = 2
    var_83 = module_0.Fieldset()
    var_84 = lambda x: x.upper()
    var_85 = module_0.Fieldset()
    var_86 = lambda x, r: x.upper()
    var_87 = module_0.Fieldset()
    var_88 = lambda x, r: x + str(r.randint(var_11, var_21))
    var_89 = module_0.Fieldset()
    var_90 = lambda x: x.upper()
    var_91 = module_0.Fieldset()
    var_92 = 'not_callable'
    var_93 = module_0.Fieldset()
    var_94 = None
    var_95 = 2



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = ';'



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 5
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.__iter__()
    var_7 = list(var_5)
    var_8 = len(var_7)
    assert var_8 == 5
    var_9 = {var_0: var_1}
    var_10 = {var_0: var_1}
    var_11 = lambda : var_10
    var_12 = 3
    var_13 = module_0.Schema(var_11, var_12)
    var_14 = 'custom_value'
    var_15 = var_13.with_context()
    var_16 = {var_0: var_1}
    var_17 = lambda : var_16
    var_18 = 2
    var_19 = module_0.Schema(var_17, var_18)
    var_20 = len(var_7)
    assert var_20 == 2
    var_21 = 'transformed'
    var_22 = True
    var_23 = {var_0: var_1}
    var_24 = lambda : var_23
    var_25 = module_0.Schema(var_24, var_12)
    var_26 = list(var_25)
    var_27 = list(var_25)
    var_28 = 'random'
    var_29 = 100
    var_30 = 42
    var_31 = {var_0: var_1}
    var_32 = lambda : var_31
    var_33 = 0
    var_34 = module_0.Schema(var_32, var_33)
    var_35 = iter(var_34)
    var_36 = next(var_35)
    var_37 = 0
    assert var_37 == 4
    var_38 = len(var_7)
    assert var_38 == 2
    var_39 = 'All tests passed!'
    var_40 = print(var_39)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'value'
    assert var_1 == 3
    var_2 = 'test'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = module_0.Schema(var_5)
    var_7 = {var_0: var_2, var_1: var_3}
    var_8 = lambda : var_7
    var_9 = module_0.Schema(var_8)
    var_10 = 2
    var_11 = {}
    var_12 = lambda : var_11
    var_13 = 0
    var_14 = module_0.Schema(var_12, var_13)
    var_15 = 'id'
    var_16 = 1
    var_17 = {var_15: var_16}
    var_18 = lambda : var_17
    var_19 = 3
    var_20 = module_0.Schema(var_18, var_19)
    var_21 = 'id'
    var_22 = 1
    var_23 = all(var_4)
    var_24 = {var_1: var_16}
    var_25 = lambda : var_24
    var_26 = module_0.Schema(var_25)
    var_27 = 'transformed'
    var_28 = True
    var_29 = lambda x: {var_27: x}
    var_30 = var_26.map(var_29)
    var_31 = {var_1: var_28}
    var_32 = lambda : var_31
    var_33 = module_0.Schema(var_32)
    var_34 = '1.0'
    var_35 = var_33.with_context()
    var_36 = 'context'
    var_37 = lambda x, ctx: {var_36: x}
    var_38 = var_33.map(var_37)
    var_39 = 'All tests passed!'
    var_40 = print(var_39)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 'test_output.pkl'
    var_2 = range(var_0)
    var_3 = [my_schema() for _ in var_2]
    var_4 = 'test_output2.pkl'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John'
    var_4 = 30
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = [var_5, var_6]



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 0
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 'id'
    var_2 = 'name'
    var_3 = 1
    var_4 = 'test'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = [var_5, var_6]



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 5
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.iterator()
    var_7 = 0
    var_8 = 1
    var_9 = var_7 + var_8
    assert var_9 == 5
    var_10 = list(var_5)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = {var_8: var_1}
    var_13 = 'num'
    var_14 = 1
    var_15 = {var_13: var_14}
    var_16 = lambda : var_15
    var_17 = 3
    var_18 = module_0.Schema(var_16, var_17)
    var_19 = lambda x: {var_13: x[var_13] + var_14}
    var_20 = var_18.map(var_19)
    var_21 = list(var_18)
    var_22 = 'index'
    var_23 = 0
    var_24 = {var_22: var_23}
    var_25 = lambda : var_24
    var_26 = 2
    var_27 = module_0.Schema(var_25, var_26)
    var_28 = lambda item, ctx: {var_22: ctx.index}
    var_29 = var_27.map(var_28)
    var_30 = list(var_27)
    var_31 = 0
    var_32 = 4
    var_33 = list(var_27)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = 'data'
    var_36 = 'test'
    var_37 = {var_35: var_36}
    var_38 = lambda : var_37
    var_39 = module_0.Schema(var_38, var_23)
    var_40 = list(var_39)
    var_41 = 'rand'
    var_42 = 100
    var_43 = 42
    var_44 = list(var_39)
    var_45 = 'All tests passed for Schema.iterator()'
    var_46 = print(var_45)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = module_0.Fieldset()
    var_1 = 'username'
    var_2 = 5
    var_3 = module_0.Fieldset()
    var_4 = module_0.Fieldset()
    var_5 = 3
    var_6 = module_0.Fieldset()
    var_7 = 'username'
    var_8 = 0
    var_9 = module_0.Fieldset()
    var_10 = 'custom_field'
    var_11 = 'custom_value'
    var_12 = lambda random, **kwargs: var_11
    var_13 = module_0.Fieldset()
    var_14 = 'alias'
    var_15 = module_0.Fieldset()
    var_16 = lambda x: x.upper()
    var_17 = module_0.Fieldset()
    var_18 = module_0.Fieldset()
    var_19 = 'person.full_name'
    var_20 = module_0.Fieldset()
    var_21 = 'full_name'
    var_22 = module_0.Fieldset()
    var_23 = 'invalid_field'
    var_24 = module_0.Fieldset()
    var_25 = 123
    var_26 = 'alias'
    var_27 = module_0.Fieldset()
    var_28 = module_0.Fieldset()
    var_29 = 'person:full_name'
    var_30 = module_0.Fieldset()
    var_31 = 'person full_name'
    var_32 = module_0.Fieldset()
    var_33 = 'person/full_name'
    var_34 = module_0.Fieldset()
    var_35 = 'person.full_name.middle'
    var_36 = 'iterations'
    var_37 = 7
    var_38 = 42
    var_39 = module_0.Fieldset()
    var_40 = module_0.Fieldset()
    var_41 = 'female'
    var_42 = module_0.Fieldset()
    var_43 = module_0.Fieldset()
    var_44 = 'test_field'
    var_45 = 'test'
    var_46 = lambda random, **kwargs: var_45
    var_47 = 'test_field'
    var_48 = module_0.Fieldset()
    var_49 = 'field1'
    var_50 = 'val1'
    var_51 = lambda random, **kwargs: var_50
    var_52 = 'field2'
    var_53 = 'val2'
    var_54 = lambda random, **kwargs: var_53
    var_55 = 'field1'
    var_56 = 'field2'
    var_57 = 100
    var_58 = module_0.Fieldset()
    var_59 = 'All tests passed!'
    var_60 = print(var_59)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.full_name'
    var_2 = lambda x: x.upper()
    var_3 = var_0.perform(var_1, var_2)
    var_4 = var_0.perform(var_1)
    var_5 = 'invalid_field'
    var_6 = var_0.perform(var_5)
    var_7 = 'custom_field'
    var_8 = var_0.perform(var_7)
    assert var_8 == 'custom'
    var_9 = 'alias'
    var_10 = var_0.perform(var_9)
    var_11 = 'person.full_name'
    var_12 = var_0.perform(var_11)
    var_13 = None
    var_14 = var_0.perform(var_13)
    var_15 = var_0.perform(var_13)
    var_16 = 'person:full_name'
    var_17 = var_0.perform(var_16)
    var_18 = 'person/full_name'
    var_19 = var_0.perform(var_18)
    var_20 = 'person full_name'
    var_21 = var_0.perform(var_20)
    var_22 = 'person.full.name'
    var_23 = var_0.perform(var_22)
    var_24 = 'female'
    var_25 = var_0.perform(var_22)
    var_26 = 'custom_with_kwargs'
    var_27 = 'test'
    var_28 = var_0.perform(var_26)
    assert var_28 == 'test'
    var_29 = var_0.unregister_handler(var_7)
    var_30 = 'custom_field'
    var_31 = var_0.perform(var_30)
    var_32 = 'handler1'
    var_33 = 'handler2'
    var_34 = var_0.unregister_all_handlers()
    var_35 = 'handler1'
    var_36 = var_0.perform(var_35)
    var_37 = 42
    var_38 = var_0.reseed(var_37)
    var_39 = var_0.perform(var_35)
    var_40 = var_0.reseed(var_37)
    var_41 = var_0.perform(var_35)
    var_42 = 'person.full_name'
    var_43 = 123
    var_44 = var_0.perform(var_43)
    var_45 = 'not_callable'
    var_46 = var_0.perform(var_43, var_45)
    var_47 = var_0.perform(var_43)
    var_48 = True
    var_49 = var_0.perform(var_43)
    var_50 = 'random_choice'
    var_51 = var_0.perform(var_50)
    var_52 = 'name'
    var_53 = var_0.perform(var_52)
    var_54 = 'invalid'
    var_55 = 'non_existent'
    var_56 = 'invalid'
    var_57 = var_0.perform(var_56)
    var_58 = 'person.full_name!'
    var_59 = var_0.perform(var_58)
    var_60 = ''
    var_61 = var_0.perform(var_60)
    var_62 = '   '
    var_63 = var_0.perform(var_62)
    var_64 = '123'
    var_65 = var_0.perform(var_64)
    var_66 = 'uuid4'
    var_67 = var_0.perform(var_66)
    var_68 = 'cryptographic.uuid4'
    var_69 = var_0.perform(var_68)
    var_70 = 'numeric.integer_number'
    var_71 = 10
    var_72 = var_0.perform(var_70)
    var_73 = var_0.perform(var_70)
    var_74 = 'numeric.integer_number'
    var_75 = 'test'
    var_76 = var_0.perform(var_74)
    var_77 = 'numeric.integer_number'
    var_78 = 1
    var_79 = 10
    var_80 = 5
    var_81 = var_0.perform(var_77)
    var_82 = 'numeric.integer_number'
    var_83 = var_0.perform(var_82)
    var_84 = var_0.perform(var_70)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = module_0.BaseField()
    var_2 = module_0.BaseField()
    var_3 = module_0.BaseField()
    var_4 = module_0.BaseField()
    var_5 = 'test'
    var_6 = var_4.handle(var_5)
    var_7 = 123
    var_8 = module_0.BaseField()



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 2
    assert var_0 == 2



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 5
    var_7 = module_0.Schema(var_5, var_6)
    var_8 = var_7.create()
    var_9 = len(var_8)
    assert var_9 == 5
    var_10 = 3
    var_11 = module_0.Schema(var_5, var_10)
    var_12 = 'transformed'
    var_13 = True
    var_14 = lambda x: {var_12: x}
    var_15 = var_11.map(var_14)
    var_16 = var_15.create()
    var_17 = module_0.Schema(var_5, var_10)
    var_18 = var_15.create()
    var_19 = 0
    var_20 = module_0.Schema(var_5, var_19)
    var_21 = var_20.create()
    var_22 = 'num'
    var_23 = 100
    var_24 = 42
    var_25 = 'All tests passed for Schema.create()'
    var_26 = print(var_25)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 'name'
    var_2 = 'John'
    var_3 = 'age'
    var_4 = 30
    var_5 = 3
    var_6 = 'transformed'
    var_7 = 2
    var_8 = 'value'
    var_9 = 'custom_field'
    var_10 = 0
    var_11 = 4
    var_12 = 'id'
    var_13 = 0
    var_14 = 42
    var_15 = 'All tests passed!'
    var_16 = print(var_15)



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = module_0.Fieldset()
    var_1 = 'username'
    var_2 = 5
    var_3 = module_0.Fieldset()
    var_4 = module_0.Fieldset()
    var_5 = 3
    var_6 = module_0.Fieldset()
    var_7 = 'username'
    var_8 = 0
    var_9 = module_0.Fieldset()
    var_10 = 'custom_field'
    var_11 = 'custom_value'
    var_12 = lambda random, **kwargs: var_11
    var_13 = module_0.Fieldset()
    var_14 = 'alias'
    var_15 = module_0.Fieldset()
    var_16 = lambda x: x.upper()
    var_17 = module_0.Fieldset()
    var_18 = module_0.Fieldset()
    var_19 = 'person.full_name'
    var_20 = module_0.Fieldset()
    var_21 = 'full_name'
    var_22 = module_0.Fieldset()
    var_23 = 'invalid_field'
    var_24 = module_0.Fieldset()
    var_25 = 'person:full_name'
    var_26 = module_0.Fieldset()
    var_27 = 'person full_name'
    var_28 = module_0.Fieldset()
    var_29 = 'person/full_name'
    var_30 = module_0.Fieldset()
    var_31 = 'person.full.name'
    var_32 = module_0.Fieldset()
    var_33 = 'my_field'
    var_34 = lambda random, **kwargs: var_11
    var_35 = module_0.Fieldset()
    var_36 = lambda random, **kwargs: var_11
    var_37 = 'custom_field'
    var_38 = module_0.Fieldset()
    var_39 = 'custom_field1'
    var_40 = 'value1'
    var_41 = lambda random, **kwargs: var_40
    var_42 = 'custom_field2'
    var_43 = 'value2'
    var_44 = lambda random, **kwargs: var_43
    var_45 = 'custom_field1'
    var_46 = 'custom_field2'
    var_47 = 42
    var_48 = module_0.Fieldset()
    var_49 = module_0.Fieldset()
    var_50 = 'female'
    var_51 = module_0.Fieldset()
    var_52 = -1
    var_53 = lambda x: x[::var_52]
    var_54 = module_0.Fieldset()
    var_55 = 'iterations'
    var_56 = 7
    var_57 = 3
    var_58 = 4
    var_59 = 'count'
    var_60 = 6
    var_61 = 8
    var_62 = module_0.Fieldset()
    var_63 = module_0.Fieldset()
    var_64 = 12
    var_65 = module_0.Fieldset()
    var_66 = 'empty_field'
    var_67 = None
    var_68 = lambda random, **kwargs: var_67
    var_69 = module_0.Fieldset()
    var_70 = 'multiply'
    var_71 = lambda random, x, y=1: x * y
    var_72 = 2
    var_73 = 4
    var_74 = module_0.Fieldset()
    var_75 = 'random_int'
    var_76 = 1
    var_77 = 10
    var_78 = lambda random, **kwargs: random.randint(var_76, var_77)
    var_79 = 20
    var_80 = module_0.Fieldset()
    var_81 = 'random_float'
    var_82 = lambda random, **kwargs: random.random()
    var_83 = module_0.Fieldset()
    var_84 = 'handler_field'
    var_85 = 'handler_value'
    var_86 = lambda random, **kwargs: var_85
    var_87 = 'alias_field'
    var_88 = module_0.Fieldset()
    var_89 = 'original'
    var_90 = 'original_value'
    var_91 = lambda random, **kwargs: var_90
    var_92 = module_0.Fieldset()
    var_93 = 'alias'
    var_94 = 123
    var_95 = module_0.Fieldset()
    var_96 = 123
    var_97 = 'username'
    var_98 = 'alias'
    var_99 = module_0.Fieldset()



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 'id'
    var_2 = 'name'
    var_3 = 1
    var_4 = 'test'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = [var_5, var_6]



# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 1
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = 'custom_value'
    var_7 = var_5.with_context()
    var_8 = 'another_value'
    var_9 = var_5.with_context()
    var_10 = 'yet_another'
    var_11 = var_5.with_context()



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 2



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = 'Test that register_handler correctly registers a new field handler.'
    var_1 = module_0.BaseField()
    var_2 = 'custom_field'
    var_3 = 'custom_value'
    var_4 = lambda random, **kwargs: var_3
    var_5 = var_1.register_handler(var_2, var_4)



# Parsed testcases at query #26
#--------------------------



def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 3
    var_7 = module_0.Schema(var_5, var_6)
    var_8 = []
    var_9 = next(var_7)
    var_10 = len(var_8)
    assert var_10 == 3
    var_11 = 2
    var_12 = module_0.Schema(var_5, var_11)
    var_13 = next(var_12)
    var_14 = next(var_12)
    var_15 = next(var_12)
    var_16 = module_0.Schema(var_5, var_11)
    var_17 = next(var_12)
    var_18 = next(var_12)
    var_19 = module_0.Schema(var_5, var_2)
    var_20 = 'bar'
    var_21 = var_19.with_context()
    var_22 = next(var_21)
    var_23 = 0
    assert var_23 == 2
    var_24 = next(var_21)
    var_25 = 'random'
    var_26 = 1000
    var_27 = 42
    var_28 = module_0.Schema(var_5, var_6, var_27)
    var_29 = range(var_6)
    var_30 = [next(var_28) for _ in var_29]
    var_31 = module_0.Schema(var_5, var_6, var_27)
    var_32 = range(var_6)
    var_33 = [next(var_31) for _ in var_32]
    var_34 = {var_15: var_2}
    var_35 = lambda : var_34
    var_36 = module_0.Schema(var_35, var_11)
    var_37 = list(var_36)
    var_38 = next(var_36)
    var_39 = iter(var_36)
    var_40 = next(var_36)
    var_41 = 'x'
    var_42 = 0
    var_43 = {var_41: var_42}
    var_44 = lambda : var_43
    var_45 = module_0.Schema(var_44, var_26)
    var_46 = 0
    var_47 = 1
    var_48 = var_46 + var_47
    assert var_48 == 1000
    var_49 = {}
    var_50 = lambda : var_49
    var_51 = module_0.Schema(var_50, var_2)
    var_52 = next(var_45)
    var_53 = {}
    var_54 = lambda : var_53
    var_55 = module_0.Schema(var_54, var_2)
    var_56 = next(var_45)
    var_57 = 'All tests passed!'
    var_58 = print(var_57)



# Parsed testcases at query #27
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 1
    assert var_1 == 3
    assert var_1 == 2
    var_2 = 2
    var_3 = False
    var_4 = 3
    var_5 = {}
    var_6 = 'All tests passed!'
    var_7 = print(var_6)



# Parsed testcases at query #28
#--------------------------



def test_case_0():
    var_0 = module_0.SchemaBuilder()
    var_1 = 'id'
    var_2 = 'name'
    var_3 = 1
    var_4 = 'test'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = lambda : var_5
    var_7 = module_0.Schema(var_6)
    var_8 = 'test_schema'
    var_9 = var_0.define(var_8, var_7)
    var_10 = 5
    var_11 = var_0.create()
    var_12 = var_11[var_8]
    var_13 = len(var_12)
    assert var_13 == 5
    var_14 = var_11[var_8]
    var_15 = var_11[var_8]
    var_16 = module_0.SchemaBuilder()
    var_17 = 'type'
    var_18 = 'user'
    var_19 = {var_17: var_18, var_1: var_3}
    var_20 = lambda : var_19
    var_21 = module_0.Schema(var_20)
    var_22 = 'product'
    var_23 = 2
    var_24 = {var_17: var_22, var_1: var_23}
    var_25 = lambda : var_24
    var_26 = module_0.Schema(var_25)
    var_27 = 'users'
    var_28 = var_16.define(var_27, var_21)
    var_29 = 'products'
    var_30 = var_16.define(var_29, var_26)
    var_31 = 3
    var_32 = var_16.create()
    var_33 = var_32[var_27]
    var_34 = len(var_33)
    assert var_34 == 3
    var_35 = var_32[var_29]
    var_36 = len(var_35)
    assert var_36 == 2
    var_37 = var_32[var_27]
    var_38 = var_32[var_29]
    var_39 = module_0.SchemaBuilder()
    var_40 = {var_1: var_3}
    var_41 = lambda : var_40
    var_42 = module_0.Schema(var_41)
    var_43 = 'indexed'
    var_44 = var_39.define(var_43, var_7)
    var_45 = var_39.create()
    var_46 = var_45[var_43]
    var_47 = len(var_46)
    assert var_47 == 3
    var_48 = module_0.SchemaBuilder()
    var_49 = {var_1: var_3}
    var_50 = lambda : var_49
    var_51 = module_0.Schema(var_50)
    var_52 = 'item_'
    var_53 = var_51.with_context()
    var_54 = 'custom'
    var_55 = var_48.define(var_54, var_7)
    var_56 = var_48.create()
    var_57 = var_56[var_54]
    var_58 = 'item_1'
    var_59 = module_0.SchemaBuilder()
    var_60 = 1
    var_61 = var_59.create()
    var_62 = module_0.SchemaBuilder()
    var_63 = var_62.create()
    var_64 = 42
    var_65 = module_0.SchemaBuilder(var_64)
    var_66 = 'random'
    var_67 = 100
    var_68 = var_65.define(var_66, var_7)
    var_69 = var_65.create()
    var_70 = module_0.SchemaBuilder(var_64)
    var_71 = var_70.define(var_66, var_26)
    var_72 = var_70.create()
    var_73 = module_0.SchemaBuilder()
    var_74 = 'User'
    var_75 = {var_60: var_3, var_61: var_74}
    var_76 = lambda : var_75
    var_77 = module_0.Schema(var_76)
    var_78 = 'user_id'
    var_79 = None
    var_80 = {var_60: var_3, var_78: var_79}
    var_81 = lambda : var_80
    var_82 = module_0.Schema(var_81)
    var_83 = var_73.define(var_27, var_77)
    var_84 = var_73.define(var_29, var_82)
    var_85 = var_73.create()
    var_86 = var_85[var_27]
    var_87 = len(var_86)
    assert var_87 == 3
    var_88 = var_85[var_29]
    var_89 = len(var_88)
    assert var_89 == 2
    var_90 = module_0.SchemaBuilder()
    var_91 = 'items'
    var_92 = []
    var_93 = {var_60: var_3, var_91: var_92}
    var_94 = lambda : var_93
    var_95 = module_0.Schema(var_94)
    var_96 = 'Product'
    var_97 = {var_60: var_3, var_61: var_96}
    var_98 = lambda : var_97
    var_99 = module_0.Schema(var_98)
    var_100 = 'orders'
    var_101 = var_90.define(var_100, var_95)
    var_102 = var_90.define(var_29, var_99)
    var_103 = var_90.create()
    var_104 = var_103[var_100]
    var_105 = len(var_104)
    assert var_105 == 2
    var_106 = var_103[var_29]
    var_107 = len(var_106)
    assert var_107 == 3
    var_108 = module_0.SchemaBuilder()
    var_109 = {var_60: var_3}
    var_110 = lambda : var_109
    var_111 = module_0.Schema(var_110)
    var_112 = 'large'
    var_113 = var_108.define(var_112, var_111)
    var_114 = 1000
    var_115 = var_108.create()
    var_116 = var_115[var_112]
    var_117 = len(var_116)
    assert var_117 == 1000
    var_118 = var_115[var_112]



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 5
    var_1 = module_0.Fieldset()
    var_2 = 'username'
    var_3 = module_0.Fieldset()
    var_4 = 0
    var_5 = module_0.Fieldset()
    var_6 = 'username'
    var_7 = 3
    var_8 = module_0.Fieldset()
    var_9 = lambda x: x.upper()
    var_10 = 'iterations'
    var_11 = 4
    var_12 = 2
    var_13 = module_0.Fieldset()
    var_14 = 'custom_field'
    var_15 = 1
    var_16 = 100
    var_17 = lambda r, **kwargs: r.randint(var_15, var_16)
    var_18 = module_0.Fieldset()
    var_19 = 'alias_field'
    var_20 = module_0.Fieldset()
    var_21 = 'invalid_field_name'
    var_22 = module_0.Fieldset()
    var_23 = 42
    var_24 = module_0.Fieldset()
    var_25 = 'person.full_name'
    var_26 = module_0.Fieldset()
    var_27 = 'full_name'
    var_28 = module_0.Fieldset()
    var_29 = 'provider.method.extra'
    var_30 = module_0.Fieldset()
    var_31 = module_0.Fieldset()
    var_32 = module_0.Fieldset()
    var_33 = 'alias'
    var_34 = 123
    var_35 = 'alias'
    var_36 = module_0.Fieldset()
    var_37 = '123invalid'
    var_38 = None
    var_39 = lambda r, **kwargs: var_38
    var_40 = module_0.Fieldset()
    var_41 = 'invalid_arity'
    var_42 = None
    var_43 = lambda r: var_42
    var_44 = module_0.Fieldset()
    var_45 = 'custom'
    var_46 = 'custom_value'
    var_47 = lambda r, **kwargs: var_46
    var_48 = 'custom'
    var_49 = module_0.Fieldset()
    var_50 = 'custom1'
    var_51 = 'value1'
    var_52 = lambda r, **kwargs: var_51
    var_53 = 'custom2'
    var_54 = 'value2'
    var_55 = lambda r, **kwargs: var_54
    var_56 = 'custom1'
    var_57 = 'custom2'
    var_58 = module_0.Fieldset()
    var_59 = module_0.Fieldset()
    var_60 = None
    var_61 = module_0.Fieldset()
    var_62 = 'person:full_name'
    var_63 = 'person/full_name'
    var_64 = 'person full_name'
    var_65 = 'All tests passed!'
    var_66 = print(var_65)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 2



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = lambda : var_2
    var_4 = 3
    var_5 = module_0.Schema(var_3, var_4)
    var_6 = var_5.iterator()
    var_7 = list(var_6)
    var_8 = 2
    var_9 = module_0.Schema(var_3, var_8)
    var_10 = 'transformed'
    var_11 = True
    var_12 = lambda x: {var_10: x}
    var_13 = var_9.map(var_12)
    var_14 = var_13.iterator()
    var_15 = list(var_14)
    var_16 = module_0.Schema(var_3, var_8)
    var_17 = 'value'
    var_18 = var_16.with_context()
    var_19 = var_18.iterator()
    var_20 = list(var_19)
    var_21 = 0
    var_22 = module_0.Schema(var_3, var_21)
    var_23 = var_22.iterator()
    var_24 = list(var_23)
    var_25 = module_0.Schema(var_3, var_8)
    var_26 = 0
    var_27 = 1
    var_28 = var_26 + var_27
    assert var_28 == 2



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = var_0.perform()
    var_2 = module_0.BaseField()
    var_3 = 'invalid'
    var_4 = var_2.perform(var_3)
    var_5 = module_0.BaseField()
    var_6 = 'person.full_name'
    var_7 = var_5.perform(var_6)
    var_8 = module_0.BaseField()
    var_9 = lambda x: x.upper()
    var_10 = var_8.perform(var_6, var_9)
    var_11 = module_0.BaseField()
    var_12 = module_0.BaseField()
    var_13 = 'custom_field'
    var_14 = var_12.perform(var_13)
    assert var_14 == 'custom'
    var_15 = module_0.BaseField()
    var_16 = 'alias'
    var_17 = var_15.perform(var_16)
    var_18 = module_0.BaseField()
    var_19 = 'alias'
    var_20 = var_18.perform(var_19)
    var_21 = module_0.BaseField()
    var_22 = 1
    var_23 = 'alias'
    var_24 = var_21.perform(var_23)
    var_25 = module_0.BaseField()
    var_26 = 'alias'
    var_27 = var_25.perform(var_26)
    var_28 = module_0.BaseField()
    var_29 = 'provider.name.extra'
    var_30 = var_28.perform(var_29)
    var_31 = module_0.BaseField()
    var_32 = 'provider-name'
    var_33 = var_31.perform(var_32)
    var_34 = module_0.BaseField()
    var_35 = 'provider name'
    var_36 = var_34.perform(var_35)
    var_37 = module_0.BaseField()
    var_38 = 'provider:name'
    var_39 = var_37.perform(var_38)
    var_40 = module_0.BaseField()
    var_41 = 'provider/name'
    var_42 = var_40.perform(var_41)
    var_43 = module_0.BaseField()
    var_44 = 'provider.name:extra'
    var_45 = var_43.perform(var_44)
    var_46 = module_0.BaseField()
    var_47 = ' provider.name '
    var_48 = var_46.perform(var_47)
    var_49 = module_0.BaseField()
    var_50 = ''
    var_51 = var_49.perform(var_50)
    var_52 = module_0.BaseField()
    var_53 = '.'
    var_54 = var_52.perform(var_53)
    var_55 = module_0.BaseField()
    var_56 = ' . '
    var_57 = var_55.perform(var_56)
    var_58 = module_0.BaseField()
    var_59 = ' : '
    var_60 = var_58.perform(var_59)
    var_61 = module_0.BaseField()
    var_62 = ' / '
    var_63 = var_61.perform(var_62)
    var_64 = module_0.BaseField()
    var_65 = ' .: '
    var_66 = var_64.perform(var_65)
    var_67 = module_0.BaseField()
    var_68 = ' . . '
    var_69 = var_67.perform(var_68)
    var_70 = module_0.BaseField()
    var_71 = ' . . '
    var_72 = var_70.perform(var_71)
    var_73 = module_0.BaseField()
    var_74 = ' : : '
    var_75 = var_73.perform(var_74)
    var_76 = module_0.BaseField()
    var_77 = ' / / '
    var_78 = var_76.perform(var_77)
    var_79 = module_0.BaseField()
    var_80 = ' .: / '
    var_81 = var_79.perform(var_80)
    var_82 = module_0.BaseField()
    var_83 = ' .: / '
    var_84 = var_82.perform(var_83)
    var_85 = module_0.BaseField()
    var_86 = ' .: : '
    var_87 = var_85.perform(var_86)
    var_88 = module_0.BaseField()
    var_89 = ' .: / '
    var_90 = var_88.perform(var_89)
    var_91 = module_0.BaseField()
    var_92 = ' .: / '
    var_93 = var_91.perform(var_92)
    var_94 = module_0.BaseField()
    var_95 = ' .: / extra'
    var_96 = var_94.perform(var_95)
    var_97 = module_0.BaseField()
    var_98 = ' .: / extra '
    var_99 = var_97.perform(var_98)
    var_100 = module_0.BaseField()
    var_101 = ' .: / extra : '
    var_102 = var_100.perform(var_101)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 3
    var_1 = 'test.pkl'
    var_2 = 'name'
    var_3 = 'value'
    var_4 = 'test'
    var_5 = 123
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_2: var_4, var_3: var_5}
    var_8 = {var_2: var_4, var_3: var_5}
    var_9 = [var_6, var_7, var_8]



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = lambda : var_4
    var_6 = 5
    var_7 = module_0.Schema(var_5, var_6)
    var_8 = var_7.create()
    var_9 = len(var_8)
    assert var_9 == 5
    var_10 = 3
    var_11 = module_0.Schema(var_5, var_10)
    var_12 = 'transformed'
    var_13 = True
    var_14 = lambda x: {var_12: x}
    var_15 = var_11.map(var_14)
    var_16 = var_11.create()
    var_17 = len(var_16)
    assert var_17 == 3
    var_18 = 2
    var_19 = module_0.Schema(var_5, var_18)
    var_20 = 'index'
    var_21 = lambda x, ctx: {var_20: x}
    var_22 = var_19.map(var_21)
    var_23 = var_19.create()
    var_24 = 0.5
    var_25 = None
    var_26 = 'data'
    var_27 = 'test'
    var_28 = {var_26: var_27}
    var_29 = 10
    var_30 = var_19.create()
    var_31 = len(var_30)
    var_32 = module_0.Schema(var_5, var_10)
    var_33 = 'value'
    var_34 = var_32.with_context()
    var_35 = 'extra'
    var_36 = lambda x, ctx: {var_35: x}
    var_37 = var_32.map(var_36)
    var_38 = var_32.create()
    var_39 = 0
    var_40 = module_0.Schema(var_5, var_39)
    var_41 = var_40.create()
    var_42 = 42
    var_43 = 'num'
    var_44 = 100
    var_45 = module_0.Schema(var_5, var_18)
    var_46 = var_45.create()
    var_47 = var_45.create()
    var_48 = len(var_46)
    assert var_48 == 2
    var_49 = 'All tests passed for Schema.create()'
    var_50 = print(var_49)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 2
    assert var_0 == 2



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = module_0.BaseField()
    var_2 = module_0.BaseField()
    var_3 = module_0.BaseField()
    var_4 = module_0.BaseField()
    var_5 = 'non_callable'
    var_6 = var_4.handle(var_5)
    var_7 = 123
    var_8 = module_0.BaseField()



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 'All tests passed for Schema.map method.'
    var_2 = print(var_1)



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 2
    var_1 = 'Pickle file was not created'
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_2: var_4, var_3: var_5}
    var_8 = [var_6, var_7]



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 3
    var_1 = 'id'
    var_2 = 1
    var_3 = 'name'
    var_4 = 'test'
    var_5 = 2
    var_6 = 'data'
    var_7 = 5
    var_8 = 0
    var_9 = 1000
    var_10 = 42
    var_11 = 'All tests passed!'
    var_12 = print(var_11)



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 5
    var_1 = 'name'
    var_2 = 'John'
    var_3 = 'age'
    var_4 = 30
    var_5 = 3
    var_6 = 2
    var_7 = 'value'
    var_8 = lambda item: {var_7: item[var_7] * var_6}
    var_9 = 'index'
    var_10 = lambda item, ctx: {var_9: ctx.index}
    var_11 = 'test'
    var_12 = 'custom'
    var_13 = None
    var_14 = 1
    var_15 = None
    var_16 = 'All tests passed!'
    var_17 = print(var_16)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 2
    assert var_0 == 2



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = module_0.BaseField()
    var_2 = module_0.BaseField()
    var_3 = module_0.BaseField()
    var_4 = module_0.BaseField()
    var_5 = 'non_callable'
    var_6 = var_4.handle(var_5)
    var_7 = 123
    var_8 = module_0.BaseField()
    var_9 = module_0.BaseField()
    var_10 = module_0.BaseField()
    var_11 = 'existing_handler'
    var_12 = var_10._handlers[var_11]
    var_13 = module_0.BaseField()
    var_14 = 'lambda_handler'
    var_15 = var_13.handle(var_14)
    var_16 = 'lambda_value'
    var_17 = lambda random, **kwargs: var_16
    var_18 = var_13._handlers[var_14]
    var_19 = callable(var_18)
    var_20 = module_0.BaseField()
    var_21 = 'static_handler'
    var_22 = var_20.handle(var_21)
    var_23 = module_0.BaseField()
    var_24 = 'instance_handler'
    var_25 = var_23.handle(var_24)
    var_26 = module_0.BaseField()
    var_27 = 'value'
    var_28 = 'partial_handler'
    var_29 = var_26.handle(var_28)
    var_30 = module_0.BaseField()
    var_31 = 'callable_class'
    var_32 = var_30.handle(var_31)
    var_33 = module_0.BaseField()
    var_34 = module_0.BaseField()
    var_35 = module_0.BaseField()
    var_36 = module_0.BaseField()
    var_37 = module_0.BaseField()
    var_38 = module_0.BaseField()
    var_39 = module_0.BaseField()
    var_40 = 'duplicate_field'
    var_41 = var_39._handlers[var_40]
    var_42 = module_0.BaseField()
    var_43 = 'random_test'
    var_44 = var_42.perform(var_43)
    var_45 = module_0.BaseField()
    var_46 = 'kwargs_test'
    var_47 = 'custom'
    var_48 = var_45.perform(var_46)
    assert var_48 == 'custom'
    var_49 = module_0.BaseField()
    var_50 = 'default_args'
    var_51 = var_49.perform(var_50)
    assert var_51 == 'default'
    var_52 = var_49.perform(var_50)
    assert var_52 == 'custom'
    var_53 = module_0.BaseField()
    var_54 = 'positional_args'
    var_55 = 'positional'
    var_56 = var_53.perform(var_54)
    assert var_56 == 'positional'
    var_57 = module_0.BaseField()
    var_58 = module_0.BaseField()



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = module_0.Fieldset()
    var_1 = 'username'
    var_2 = 5
    var_3 = module_0.Fieldset()
    var_4 = 0
    var_5 = module_0.Fieldset()
    var_6 = 'username'
    var_7 = -1
    var_8 = module_0.Fieldset()
    var_9 = 'username'
    var_10 = 1
    var_11 = module_0.Fieldset()
    var_12 = 100
    var_13 = module_0.Fieldset()
    var_14 = 10
    var_15 = module_0.Fieldset()
    var_16 = 'email'
    var_17 = module_0.Fieldset()
    var_18 = 'password'
    var_19 = module_0.Fieldset()
    var_20 = lambda x: x.upper()
    var_21 = module_0.Fieldset()
    var_22 = lambda x, r: x + str(r.randint(var_10, var_14))
    var_23 = module_0.Fieldset()
    var_24 = lambda x, r: x + str(r.randint(var_10, var_14))
    var_25 = module_0.Fieldset()
    var_26 = lambda x, r: x + str(r.randint(var_10, var_14))
    var_27 = module_0.Fieldset()
    var_28 = lambda x, r: x + str(r.randint(var_10, var_14))
    var_29 = module_0.Fieldset()
    var_30 = lambda x, r: x + str(r.randint(var_10, var_14))
    var_31 = module_0.Fieldset()
    var_32 = 'custom_field'
    var_33 = 'length'
    var_34 = lambda r, **kwargs: r.randstr(length=kwargs.get(var_33, var_14))
    var_35 = module_0.Fieldset()
    var_36 = module_0.Fieldset()
    var_37 = module_0.Fieldset()
    var_38 = 42
    var_39 = module_0.Fieldset()
    var_40 = lambda r, **kwargs: r.randstr(length=kwargs.get(var_33, var_14))
    var_41 = 'custom_field'
    var_42 = 10
    var_43 = module_0.Fieldset()
    var_44 = lambda r, **kwargs: r.randstr(length=kwargs.get(var_33, var_14))
    var_45 = 'custom_field'
    var_46 = 10
    var_47 = 'All tests passed!'
    var_48 = print(var_47)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 2



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'Alice'
    var_4 = 0
    var_5 = 'users'
    var_6 = None
    var_7 = module_0.SchemaContext(var_4)
    var_8 = 'users'
    var_9 = var_7.pick_from(var_8)
    var_10 = "Schema 'nonexistent' not found"
    var_11 = 'nonexistent'



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = module_0.BaseField()
    var_1 = 'person.full_name'
    var_2 = var_0.perform(var_1)
    var_3 = module_0.BaseField()
    var_4 = lambda x: x.upper()
    var_5 = var_3.perform(var_1, var_4)
    var_6 = module_0.BaseField()
    var_7 = 'invalid_field'
    var_8 = var_6.perform(var_7)
    var_9 = module_0.BaseField()
    var_10 = 'alias'
    var_11 = var_9.perform(var_10)
    var_12 = module_0.BaseField()
    var_13 = 'custom_field'
    var_14 = 'custom_value'
    var_15 = lambda random, **kwargs: var_14
    var_16 = var_12.register_handler(var_13, var_15)
    var_17 = var_12.perform(var_13)
    assert var_17 == 'custom_value'
    var_18 = module_0.BaseField()
    var_19 = 1
    var_20 = 10
    var_21 = lambda result, random: result + str(random.randint(var_19, var_20))
    var_22 = var_18.perform(var_7, var_21)
    var_23 = len(var_22)
    var_24 = module_0.BaseField()
    var_25 = lambda result: result.upper()
    var_26 = var_24.perform(var_7, var_25)
    var_27 = module_0.BaseField()
    var_28 = None
    var_29 = var_27.perform(var_28)
    var_30 = module_0.BaseField()
    var_31 = ''
    var_32 = var_30.perform(var_31)
    var_33 = module_0.BaseField()
    var_34 = 'provider.name.extra'
    var_35 = var_33.perform(var_34)
    var_36 = module_0.BaseField()
    var_37 = 'provider:name'
    var_38 = var_36.perform(var_37)
    var_39 = module_0.BaseField()
    var_40 = 'provider/name'
    var_41 = var_39.perform(var_40)
    var_42 = module_0.BaseField()
    var_43 = 'provider name'
    var_44 = var_42.perform(var_43)
    var_45 = module_0.BaseField()
    var_46 = 'female'
    var_47 = var_45.perform(var_34)
    var_48 = module_0.BaseField()
    var_49 = 'person.full_name'
    var_50 = 'value'
    var_51 = var_48.perform(var_49)
    var_52 = module_0.BaseField()
    var_53 = module_0.BaseField()
    var_54 = 123
    var_55 = module_0.BaseField()
    var_56 = module_0.BaseField()
    var_57 = module_0.BaseField()
    var_58 = module_0.BaseField()
    var_59 = module_0.BaseField()
    var_60 = module_0.BaseField()
    var_61 = module_0.BaseField()
    var_62 = module_0.BaseField()
    var_63 = module_0.BaseField()
    var_64 = module_0.BaseField()
    var_65 = 456
    var_66 = module_0.BaseField()
    var_67 = module_0.BaseField()
    var_68 = module_0.BaseField()
    var_69 = module_0.BaseField()
    var_70 = ''
    var_71 = module_0.BaseField()
    var_72 = module_0.BaseField()
    var_73 = module_0.BaseField()
    var_74 = ' '
    var_75 = module_0.BaseField()
    var_76 = module_0.BaseField()
    var_77 = module_0.BaseField()
    var_78 = '!@#$%^&*()'
    var_79 = module_0.BaseField()
    var_80 = module_0.BaseField()
    var_81 = module_0.BaseField()
    var_82 = 'alias_unicode'
    var_83 = module_0.BaseField()
    var_84 = 'person.full_name_unicode'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = 2



