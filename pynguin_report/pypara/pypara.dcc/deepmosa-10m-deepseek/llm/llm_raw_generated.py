####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_register_success():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="TestDCC",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    registry.register(dcc)
    assert registry.find("TestDCC") == dcc
    assert registry.find("TestAlt") == dcc

def test_register_duplicate_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="TestDCC",
        altnames=set(),
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    registry.register(dcc1)
    dcc2 = DCC(
        name="TestDCC",
        altnames=set(),
        currencies={Currency("EUR")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.6")
    )
    try:
        registry.register(dcc2)
        assert False
    except TypeError:
        assert True

def test_register_duplicate_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="TestDCC1",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    registry.register(dcc1)
    dcc2 = DCC(
        name="TestDCC2",
        altnames={"TestAlt"},
        currencies={Currency("EUR")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.6")
    )
    try:
        registry.register(dcc2)
        assert False
    except TypeError:
        assert True


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_register_new_dcc():
    dcc = DCC(name="TestDCC", altnames={"TestAlt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"))
    registry = DCCRegistryMachinery()
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("TestAlt") == dcc

def test_register_duplicate_main_name():
    dcc1 = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"))
    dcc2 = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.2"))
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

def test_register_duplicate_alt_name():
    dcc1 = DCC(name="TestDCC1", altnames={"TestAlt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"))
    dcc2 = DCC(name="TestDCC2", altnames={"TestAlt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.2"))
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass

def test_register_duplicate_name_in_alt_names():
    dcc1 = DCC(name="TestDCC1", altnames={"TestDCC2"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"))
    dcc2 = DCC(name="TestDCC2", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.2"))
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_act_365_l_with_non_leap_year():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 12, 31)
    expected = Decimal("0.99726027397260")
    result = dcfc_act_365_l(start, asof, asof)
    assert result == expected

def test_dcfc_act_365_l_with_leap_year():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    expected = Decimal("0.99726775956284")
    result = dcfc_act_365_l(start, asof, asof)
    assert result == expected

def test_dcfc_act_365_l_with_single_day_in_non_leap_year():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 1, 2)
    expected = Decimal("0.00273972602740")
    result = dcfc_act_365_l(start, asof, asof)
    assert result == expected

def test_dcfc_act_365_l_with_single_day_in_leap_year():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    expected = Decimal("0.00273224043716")
    result = dcfc_act_365_l(start, asof, asof)
    assert result == expected

def test_dcfc_act_365_l_with_same_day():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 1, 1)
    expected = Decimal("0.0")
    result = dcfc_act_365_l(start, asof, asof)
    assert result == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #4
#--------------------------

```python
def test_dcfc_act_365_a():
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    result1 = dcfc_act_365_a(start1, asof1, asof1)
    assert round(result1, 14) == Decimal('0.16986301369863')

    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    result2 = dcfc_act_365_a(start2, asof2, asof2)
    assert round(result2, 14) == Decimal('0.17213114754098')

    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    result3 = dcfc_act_365_a(start3, asof3, asof3)
    assert round(result3, 14) == Decimal('1.08196721311475')

    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    result4 = dcfc_act_365_a(start4, asof4, asof4)
    assert round(result4, 14) == Decimal('1.32513661202186')

    start5 = datetime.date(2020, 1, 1)
    asof5 = datetime.date(2020, 12, 31)
    result5 = dcfc_act_365_a(start5, asof5, asof5)
    assert round(result5, 14) == Decimal('0.99726775956284')

    start6 = datetime.date(2019, 1, 1)
    asof6 = datetime.date(2019, 12, 31)
    result6 = dcfc_act_365_a(start6, asof6, asof6)
    assert round(result6, 14) == Decimal('0.99726027397260')


# LLM-generated content at query #5
#--------------------------

```python
def test_dcfc_30_e_360_basic_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start, asof, end)
    assert round(result, 14) == Decimal('0.16666666666667')

def test_dcfc_30_e_360_leap_year_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start, asof, end)
    assert round(result, 14) == Decimal('0.16944444444444')

def test_dcfc_30_e_360_31st_day_adjustment():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start, asof, end)
    assert round(result, 14) == Decimal('1.08333333333333')

def test_dcfc_30_e_360_multi_year_case():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start, asof, end)
    assert round(result, 14) == Decimal('1.33055555555556')

def test_dcfc_30_e_360_same_date():
    start = datetime.date(2023, 1, 15)
    asof = datetime.date(2023, 1, 15)
    end = datetime.date(2023, 1, 15)
    result = dcfc_30_e_360(start, asof, end)
    assert result == Decimal('0')

def test_dcfc_30_e_360_31st_asof_adjustment():
    start = datetime.date(2023, 1, 15)
    asof = datetime.date(2023, 3, 31)
    end = datetime.date(2023, 3, 31)
    result = dcfc_30_e_360(start, asof, end)
    assert round(result, 14) == Decimal('0.22222222222222')


# LLM-generated content at query #6
#--------------------------

```python
def test_dcfc_30_360_isda_basic_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_isda_leap_year_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_isda_31_day_month_case():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_isda_multi_year_case():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('1.33333333333333')

def test_dcfc_30_360_isda_start_31_adjustment():
    start = datetime.date(2007, 1, 31)
    asof = datetime.date(2007, 2, 28)
    end = datetime.date(2007, 2, 28)
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('0.08333333333333')

def test_dcfc_30_360_isda_asof_31_adjustment():
    start = datetime.date(2007, 1, 30)
    asof = datetime.date(2007, 2, 31)
    end = datetime.date(2007, 2, 31)
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('0.08333333333333')


# LLM-generated content at query #7
#--------------------------

def test_dcfc_act_365_a_basic_calculation():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    assert round(result, 14) == Decimal('0.16986301369863')

def test_dcfc_act_365_a_leap_year_calculation():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    assert round(result, 14) == Decimal('0.17213114754098')

def test_dcfc_act_365_a_multi_year_calculation():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    assert round(result, 14) == Decimal('1.08196721311475')

def test_dcfc_act_365_a_long_period_calculation():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    assert round(result, 14) == Decimal('1.32513661202186')

def test_dcfc_act_365_a_same_day():
    date = datetime.date(2020, 1, 1)
    result = dcfc_act_365_a(date, date, date)
    assert result == Decimal('0')

def test_dcfc_act_365_a_one_day():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    assert result == Decimal('1') / Decimal(366)


# LLM-generated content at query #8
#--------------------------

```python
def test_calculate_daily_fraction_valid_dates():
    dcc = DCC(name="ACT/ACT", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.01"))
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 2)
    end = datetime.date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal("0.01")

def test_calculate_daily_fraction_asof_before_start():
    dcc = DCC(name="ACT/ACT", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.01"))
    start = datetime.date(2023, 1, 2)
    asof = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal("0.01")

def test_calculate_daily_fraction_asof_equals_start():
    dcc = DCC(name="ACT/ACT", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.01"))
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal("0.00")

def test_calculate_daily_fraction_asof_equals_end():
    dcc = DCC(name="ACT/ACT", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.01"))
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 3)
    end = datetime.date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal("0.01")

def test_calculate_daily_fraction_asof_after_end():
    dcc = DCC(name="ACT/ACT", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.01"))
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 4)
    end = datetime.date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal("0.00")


# LLM-generated content at query #9
#--------------------------

```python
def test_last_payment_date_annual_frequency():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    result = _last_payment_date(start, asof, 1)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_same_year_annual_frequency():
    start = datetime.date(2015, 1, 1)
    asof = datetime.date(2015, 12, 31)
    result = _last_payment_date(start, asof, 1)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_semi_annual_frequency():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    result = _last_payment_date(start, asof, 2)
    assert result == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_frequency_mid_year():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 8, 31)
    result = _last_payment_date(start, asof, 2)
    assert result == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_frequency_early_year():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 4, 30)
    result = _last_payment_date(start, asof, 2)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_annual_frequency_mid_year_start():
    start = datetime.date(2014, 6, 1)
    asof = datetime.date(2015, 4, 30)
    result = _last_payment_date(start, asof, 1)
    assert result == datetime.date(2014, 6, 1)

def test_last_payment_date_quarterly_frequency():
    start = datetime.date(2008, 7, 7)
    asof = datetime.date(2015, 10, 6)
    result = _last_payment_date(start, asof, 4)
    assert result == datetime.date(2015, 7, 7)

def test_last_payment_date_annual_frequency_december_start():
    start = datetime.date(2014, 12, 9)
    asof = datetime.date(2015, 12, 4)
    result = _last_payment_date(start, asof, 1)
    assert result == datetime.date(2014, 12, 9)

def test_last_payment_date_semi_annual_frequency_december_start():
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2016, 1, 6)
    result = _last_payment_date(start, asof, 2)
    assert result == datetime.date(2015, 12, 15)

def test_last_payment_date_semi_annual_frequency_december_end():
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2015, 12, 31)
    result = _last_payment_date(start, asof, 2)
    assert result == datetime.date(2015, 12, 15)


# LLM-generated content at query #10
#--------------------------

```python
def test_30_360_isda_asof_day_31_adjustment():
    start = datetime.date(2023, 10, 30)
    asof = datetime.date(2023, 11, 31)
    end = datetime.date(2023, 11, 31)
    result = dcfc_30_360_isda(start, asof, end)
    assert result == Decimal('0.08333333333333')


# LLM-generated content at query #11
#--------------------------

```python
def test_constructor_initializes_buffers_correctly():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #12
#--------------------------

```python
def test_register_raises_typeerror_when_dcc_name_already_registered():
    dcc1 = DCC(name="Act/Act", altnames=set(), currencies=set(), calculate_fraction_method=lambda x, y, z, w: Decimal(1))
    dcc2 = DCC(name="Act/Act", altnames=set(), currencies=set(), calculate_fraction_method=lambda x, y, z, w: Decimal(2))
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    registry.register(dcc2)


# LLM-generated content at query #13
#--------------------------

```python
def test_last_payment_date_eom_false():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = None
    result = _last_payment_date(start, asof, frequency, eom)
    assert result != datetime.date(2015, 12, 31)


# LLM-generated content at query #14
#--------------------------

```python
def test_dcfc_30_360_german_with_example1():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_german_with_example2():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_german_with_example3():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_german_with_example4():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33055555555556')

def test_dcfc_30_360_german_with_31st_start_day():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.08333333333333')

def test_dcfc_30_360_german_with_feb_end_start_day():
    start = datetime.date(2023, 2, 28)
    asof = datetime.date(2023, 3, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.08333333333333')

def test_dcfc_30_360_german_with_31st_asof_day():
    start = datetime.date(2023, 1, 15)
    asof = datetime.date(2023, 1, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.04444444444444')

def test_dcfc_30_360_german_with_feb_end_asof_day():
    start = datetime.date(2023, 1, 15)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 3, 15)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('0.11944444444444')


# LLM-generated content at query #15
#--------------------------

```python
def test__buffer_main_is_empty_dict_after_initialization():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}


# LLM-generated content at query #16
#--------------------------

```python
def test_find_existing_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", [])
    registry.register(dcc)
    result = registry.find("Act/Act")
    assert result == dcc

def test_find_non_existing_dcc():
    registry = DCCRegistryMachinery()
    result = registry.find("NonExistingDCC")
    assert result is None

def test_find_stripped_and_uppercased_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", [])
    registry.register(dcc)
    result = registry.find(" act/act ")
    assert result == dcc

def test_find_alternative_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["Actual/Actual"])
    registry.register(dcc)
    result = registry.find("Actual/Actual")
    assert result == dcc

def test_find_stripped_and_uppercased_alternative_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["Actual/Actual"])
    registry.register(dcc)
    result = registry.find(" actual/actual ")
    assert result == dcc


# LLM-generated content at query #17
#--------------------------

Here are the unit tests for the `dcfc_act_act` function:


# LLM-generated content at query #18
#--------------------------

```python
def test_has_leap_day_with_leap_day_in_range():
    start = date(2020, 1, 1)
    end = date(2020, 12, 31)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_without_leap_day_in_range():
    start = date(2021, 1, 1)
    end = date(2021, 12, 31)
    result = _has_leap_day(start, end)
    assert result == False

def test_has_leap_day_with_range_across_leap_year():
    start = date(2019, 1, 1)
    end = date(2021, 12, 31)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_with_range_before_leap_year():
    start = date(2017, 1, 1)
    end = date(2019, 12, 31)
    result = _has_leap_day(start, end)
    assert result == False

def test_has_leap_day_with_range_after_leap_year():
    start = date(2021, 1, 1)
    end = date(2023, 12, 31)
    result = _has_leap_day(start, end)
    assert result == False

def test_has_leap_day_with_single_day_range_on_leap_day():
    start = date(2020, 2, 29)
    end = date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_with_single_day_range_not_on_leap_day():
    start = date(2020, 2, 28)
    end = date(2020, 2, 28)
    result = _has_leap_day(start, end)
    assert result == False


# LLM-generated content at query #19
#--------------------------

```python
def test_dcfc_act_act_with_invalid_date_range():
    start = datetime.date(2008, 2, 28)
    asof = datetime.date(2007, 12, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start, asof, end)
    assert result == Decimal('0')


# LLM-generated content at query #20
#--------------------------

def test_dcfc_30_360_us_basic_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start, asof, asof)
    assert result == Decimal('0.1666666666666666666666666667')

def test_dcfc_30_360_us_leap_year_case():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start, asof, asof)
    assert result == Decimal('0.1694444444444444444444444444')

def test_dcfc_30_360_us_month_end_case():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start, asof, asof)
    assert result == Decimal('1.083333333333333333333333333')

def test_dcfc_30_360_us_long_period_case():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start, asof, asof)
    assert result == Decimal('1.333333333333333333333333333')

def test_dcfc_30_360_us_same_date():
    date = datetime.date(2020, 1, 1)
    result = dcfc_30_360_us(date, date, date)
    assert result == Decimal('0')

def test_dcfc_30_360_us_31_to_30_adjustment():
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 28)
    result = dcfc_30_360_us(start, asof, asof)
    assert result == Decimal('0.08333333333333333333333333333')

def test_dcfc_30_360_us_both_dates_last_day_of_month():
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 29)
    result = dcfc_30_360_us(start, asof, asof)
    assert result == Decimal('0.08333333333333333333333333333')


# LLM-generated content at query #21
#--------------------------

```python
def test_coupon_calculation():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("1000"), Currency.USD)
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("25"), Currency.USD)

def test_coupon_calculation_with_eom():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("1000"), Currency.USD)
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)
    freq = 2
    eom = 15
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("25"), Currency.USD)

def test_coupon_calculation_with_different_frequency():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.25"))
    principal = Money(Decimal("1000"), Currency.USD)
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)
    freq = 4
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("12.5"), Currency.USD)

def test_coupon_calculation_with_zero_fraction():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0"))
    principal = Money(Decimal("1000"), Currency.USD)
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("0"), Currency.USD)


# LLM-generated content at query #22
#--------------------------

```python
def test_register_raises_error_when_name_already_registered():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Act/Act", altnames={"Actual/Actual"}, currencies={"USD"}, calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc1)
    dcc2 = DCC(name="Act/Act", altnames={"Actual/360"}, currencies={"EUR"}, calculate_fraction_method=lambda s, a, e, f: Decimal("0.6"))
    try:
        registry.register(dcc2)
    except TypeError as e:
        assert str(e) == "Day count convention 'Act/Act' is already registered"


# LLM-generated content at query #23
#--------------------------

```python
def test_calculate_daily_fraction_asof_minus_1_less_than_start():
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 12, 31)
    freq = Decimal("1.0")
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("1.0"))
    result = dcc.calculate_daily_fraction(start, asof, end, freq)
    assert result == Decimal("1.0")


# LLM-generated content at query #24
#--------------------------

```
def test_find_returns_dcc_when_name_matches_case_insensitive():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", [], None, None)
    registry.register(dcc)
    result = registry.find("act/act")
    assert result == dcc

def test_find_returns_dcc_when_name_matches_with_whitespace():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", [], None, None)
    registry.register(dcc)
    result = registry.find("  Act/Act  ")
    assert result == dcc

def test_find_returns_dcc_when_name_matches_case_insensitive_with_whitespace():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", [], None, None)
    registry.register(dcc)
    result = registry.find("  act/act  ")
    assert result == dcc

def test_find_returns_none_when_name_not_registered():
    registry = DCCRegistryMachinery()
    result = registry.find("NonExistent")
    assert result is None


# LLM-generated content at query #25
#--------------------------

```python
def test_has_leap_day_with_leap_day_in_range():
    start_date = date(2020, 1, 1)
    end_date = date(2020, 12, 31)
    assert _has_leap_day(start_date, end_date) == True

def test_has_leap_day_with_leap_day_at_start():
    start_date = date(2020, 2, 29)
    end_date = date(2020, 12, 31)
    assert _has_leap_day(start_date, end_date) == True

def test_has_leap_day_with_leap_day_at_end():
    start_date = date(2020, 1, 1)
    end_date = date(2020, 2, 29)
    assert _has_leap_day(start_date, end_date) == True

def test_has_leap_day_with_leap_day_in_multiple_years():
    start_date = date(2020, 1, 1)
    end_date = date(2024, 12, 31)
    assert _has_leap_day(start_date, end_date) == True


# LLM-generated content at query #26
#--------------------------

Here are the test cases for the `dcfc_act_365_l` function:


# LLM-generated content at query #27
#--------------------------

```python
def test_dcfc_30_360_german_predicate_evaluates_to_true():
    start_date = datetime.date(2007, 12, 31)
    asof_date = datetime.date(2008, 2, 29)
    end_date = datetime.date(2008, 2, 29)
    dcfc_30_360_german(start_date, asof_date, end_date)


# LLM-generated content at query #28
#--------------------------

```python
def test_has_leap_day():
    start = date(2020, 1, 1)
    end = date(2020, 12, 31)
    result = _has_leap_day(start, end)
    assert result == True


# LLM-generated content at query #29
#--------------------------

```python
def test_dcfc_30_e_360_with_start_day_not_31():
    start = datetime.date(2007, 12, 30)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    assert result == Decimal('0.16666666666667')


# LLM-generated content at query #30
#--------------------------

```python
def test_calculate_daily_fraction_asof_minus_1_not_less_than_start():
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 2)
    end = datetime.date(2023, 1, 3)
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda x, y, z, w: Decimal(1))
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result != ZERO


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_30_360_isda_example_1():
    start = date(2007, 12, 28)
    asof = date(2008, 2, 28)
    result = round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14)
    assert result == Decimal('0.16666666666667')

def test_dcfc_30_360_isda_example_2():
    start = date(2007, 12, 28)
    asof = date(2008, 2, 29)
    result = round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14)
    assert result == Decimal('0.16944444444444')

def test_dcfc_30_360_isda_example_3():
    start = date(2007, 10, 31)
    asof = date(2008, 11, 30)
    result = round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14)
    assert result == Decimal('1.08333333333333')

def test_dcfc_30_360_isda_example_4():
    start = date(2008, 2, 1)
    asof = date(2009, 5, 31)
    result = round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14)
    assert result == Decimal('1.33333333333333')

def test_dcfc_30_360_isda_start_day_31():
    start = date(2023, 1, 31)
    asof = date(2023, 2, 28)
    result = round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14)
    assert result == Decimal('0.07777777777778')

def test_dcfc_30_360_isda_asof_day_31():
    start = date(2023, 1, 30)
    asof = date(2023, 2, 31)
    result = round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14)
    assert result == Decimal('0.08333333333333')

def test_dcfc_30_360_isda_leap_year():
    start = date(2020, 2, 28)
    asof = date(2020, 2, 29)
    result = round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14)
    assert result == Decimal('0.00277777777778')

def test_dcfc_30_360_isda_year_change():
    start = date(2022, 12, 30)
    asof = date(2023, 1, 30)
    result = round(dcfc_30_360_isda(start=start, asof=asof, end=asof), 14)
    assert result == Decimal('0.08333333333333')


# LLM-generated content at query #2
#--------------------------

```python
def test_coupon_calculation():
    dcc = DCC(name="30/360", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)
    freq = Decimal("2")
    eom = None
    assert dcc.coupon(principal, rate, start, asof, end, freq, eom) == Money(25, Currency("USD"))

def test_coupon_calculation_with_eom():
    dcc = DCC(name="30/360", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)
    freq = Decimal("2")
    eom = 31
    assert dcc.coupon(principal, rate, start, asof, end, freq, eom) == Money(25, Currency("USD"))

def test_coupon_calculation_with_different_frequency():
    dcc = DCC(name="30/360", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.25"))
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)
    freq = Decimal("4")
    eom = None
    assert dcc.coupon(principal, rate, start, asof, end, freq, eom) == Money(12.5, Currency("USD"))

def test_coupon_calculation_with_zero_fraction():
    dcc = DCC(name="30/360", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.0"))
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)
    freq = Decimal("2")
    eom = None
    assert dcc.coupon(principal, rate, start, asof, end, freq, eom) == Money(0, Currency("USD"))


# LLM-generated content at query #3
#--------------------------

```python
def test_find_existing_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    assert registry.find("Act/Act") == dcc

def test_find_non_existing_dcc():
    registry = DCCRegistryMachinery()
    assert registry.find("NonExistent") is None

def test_find_with_stripped_and_uppercase():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    assert registry.find("act/act") == dcc

def test_find_with_alternative_names():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=["Actual/Actual"])
    registry.register(dcc)
    assert registry.find("Actual/Actual") == dcc

def test_find_with_stripped_and_uppercase_alternative_names():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=["Actual/Actual"])
    registry.register(dcc)
    assert registry.find("actual/actual") == dcc


# LLM-generated content at query #4
#--------------------------

```python
def test_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #5
#--------------------------

```python
def test_buffer_main_is_empty_dict():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}

def test_buffer_altn_is_empty_dict():
    registry = DCCRegistryMachinery()
    assert registry._buffer_altn == {}


# LLM-generated content at query #6
#--------------------------

```python
def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #7
#--------------------------

```python
def test_is_last_day_of_month_true():
    date = Date(2023, 1, 31)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_false():
    date = Date(2023, 1, 30)
    assert _is_last_day_of_month(date) == False

def test_is_last_day_of_month_february_non_leap_year():
    date = Date(2023, 2, 28)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_february_leap_year():
    date = Date(2024, 2, 29)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_april():
    date = Date(2023, 4, 30)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_june():
    date = Date(2023, 6, 30)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_september():
    date = Date(2023, 9, 30)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_november():
    date = Date(2023, 11, 30)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_december():
    date = Date(2023, 12, 31)
    assert _is_last_day_of_month(date) == True


# LLM-generated content at query #8
#--------------------------

```
def test_register_new_dcc():
    dcc = DCC(
        name="TestDCC",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"),
    )
    registry = DCCRegistryMachinery()
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("TestAlt") == dcc

def test_register_duplicate_main_name():
    dcc1 = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"),
    )
    dcc2 = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"),
    )
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError:
        assert True

def test_register_duplicate_alt_name():
    dcc1 = DCC(
        name="TestDCC1",
        altnames={"TestAlt"},
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"),
    )
    dcc2 = DCC(
        name="TestDCC2",
        altnames={"TestAlt"},
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"),
    )
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError:
        assert True

def test_register_duplicate_main_name_as_alt_name():
    dcc1 = DCC(
        name="TestDCC1",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"),
    )
    dcc2 = DCC(
        name="TestDCC2",
        altnames={"TestDCC1"},
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"),
    )
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #9
#--------------------------

```python
def test_last_payment_date_annual_frequency():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    result = _last_payment_date(start, asof, 1)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_same_year_annual_frequency():
    start = datetime.date(2015, 1, 1)
    asof = datetime.date(2015, 12, 31)
    result = _last_payment_date(start, asof, 1)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_semi_annual_frequency():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    result = _last_payment_date(start, asof, 2)
    assert result == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_frequency_mid_year():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 8, 31)
    result = _last_payment_date(start, asof, 2)
    assert result == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_frequency_early_year():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 4, 30)
    result = _last_payment_date(start, asof, 2)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_annual_frequency_mid_year_start():
    start = datetime.date(2014, 6, 1)
    asof = datetime.date(2015, 4, 30)
    result = _last_payment_date(start, asof, 1)
    assert result == datetime.date(2014, 6, 1)

def test_last_payment_date_quarterly_frequency():
    start = datetime.date(2008, 7, 7)
    asof = datetime.date(2015, 10, 6)
    result = _last_payment_date(start, asof, 4)
    assert result == datetime.date(2015, 7, 7)

def test_last_payment_date_annual_frequency_late_year():
    start = datetime.date(2014, 12, 9)
    asof = datetime.date(2015, 12, 4)
    result = _last_payment_date(start, asof, 1)
    assert result == datetime.date(2014, 12, 9)

def test_last_payment_date_semi_annual_frequency_december_start():
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2016, 1, 6)
    result = _last_payment_date(start, asof, 2)
    assert result == datetime.date(2015, 12, 15)

def test_last_payment_date_semi_annual_frequency_end_of_year():
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2015, 12, 31)
    result = _last_payment_date(start, asof, 2)
    assert result == datetime.date(2015, 12, 15)


# LLM-generated content at query #10
#--------------------------

```python
def test_dcfc_30_360_us():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #11
#--------------------------

```python
def test_coupon_with_annual_frequency():
    dcc = DCC(
        name="30/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"),
    )
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 6, 30)
    end = datetime.date(2022, 1, 1)
    freq = 1
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result.amount == Decimal("25.00")
    assert result.currency == "USD"


def test_coupon_with_semiannual_frequency():
    dcc = DCC(
        name="30/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.25"),
    )
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 9, 30)
    end = datetime.date(2021, 1, 1)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result.amount == Decimal("12.50")
    assert result.currency == "USD"


def test_coupon_with_quarterly_frequency():
    dcc = DCC(
        name="30/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.125"),
    )
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 4, 15)
    end = datetime.date(2020, 7, 1)
    freq = 4
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result.amount == Decimal("6.25")
    assert result.currency == "USD"


def test_coupon_with_eom_adjustment():
    dcc = DCC(
        name="30/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"),
    )
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2021, 6, 30)
    end = datetime.date(2022, 1, 31)
    freq = 1
    eom = 31
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result.amount == Decimal("25.00")
    assert result.currency == "USD"


def test_coupon_with_asof_equal_to_start():
    dcc = DCC(
        name="30/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.0"),
    )
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = 1
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result.amount == Decimal("0.00")
    assert result.currency == "USD"


def test_coupon_with_asof_equal_to_end():
    dcc = DCC(
        name="30/360",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("1.0"),
    )
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = 1
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result.amount == Decimal("50.00")
    assert result.currency == "USD"


# LLM-generated content at query #12
#--------------------------

```python
def test_last_payment_date():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 7, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2) == datetime.date(2015, 7, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2014, 6, 1)
    assert _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4) == datetime.date(2015, 7, 7)
    assert _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1) == datetime.date(2014, 12, 9)
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2) == datetime.date(2015, 12, 15)
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 12, 15)


# LLM-generated content at query #13
#--------------------------

```python
def test_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #14
#--------------------------

```python
def test_dcfc_30_e_plus_360_example_1():
    start = date(2007, 12, 28)
    asof = date(2008, 2, 28)
    result = dcfc_30_e_plus_360(start, asof, asof)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

def test_dcfc_30_e_plus_360_example_2():
    start = date(2007, 12, 28)
    asof = date(2008, 2, 29)
    result = dcfc_30_e_plus_360(start, asof, asof)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

def test_dcfc_30_e_plus_360_example_3():
    start = date(2007, 10, 31)
    asof = date(2008, 11, 30)
    result = dcfc_30_e_plus_360(start, asof, asof)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_e_plus_360_example_4():
    start = date(2008, 2, 1)
    asof = date(2009, 5, 31)
    result = dcfc_30_e_plus_360(start, asof, asof)
    expected = Decimal('1.33333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_e_plus_360_with_start_day_31():
    start = date(2007, 12, 31)
    asof = date(2008, 2, 28)
    result = dcfc_30_e_plus_360(start, asof, asof)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

def test_dcfc_30_e_plus_360_with_asof_day_31():
    start = date(2007, 12, 28)
    asof = date(2008, 2, 29)
    result = dcfc_30_e_plus_360(start, asof, asof)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected


# LLM-generated content at query #15
#--------------------------

```python
def test_init_registry_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #16
#--------------------------

def test_dcfc_30_360_german_with_regular_dates():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    assert round(result, 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_german_with_leap_year_date():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    assert round(result, 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_german_with_31st_day_start():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    assert round(result, 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_german_with_feb_start():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    assert round(result, 14) == Decimal('1.33055555555556')

def test_dcfc_30_360_german_with_end_not_equal_to_asof():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 3, 31)
    result = dcfc_30_360_german(start, asof, end)
    assert round(result, 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_german_with_last_day_of_feb_start():
    start = datetime.date(2008, 2, 29)
    asof = datetime.date(2008, 3, 31)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    assert round(result, 14) == Decimal('0.08333333333333')


# LLM-generated content at query #17
#--------------------------

```python
def test_dcfc_30_e_360():
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start1, asof1, asof1), 14) == Decimal('0.16666666666667')

    start2 = datetime.date(2007, 12, 28)
    asof2 = datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start2, asof2, asof2), 14) == Decimal('0.16944444444444')

    start3 = datetime.date(2007, 10, 31)
    asof3 = datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start3, asof3, asof3), 14) == Decimal('1.08333333333333')

    start4 = datetime.date(2008, 2, 1)
    asof4 = datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start4, asof4, asof4), 14) == Decimal('1.33055555555556')

    start5 = datetime.date(2020, 1, 31)
    asof5 = datetime.date(2020, 3, 31)
    assert round(dcfc_30_e_360(start5, asof5, asof5), 14) == Decimal('0.16666666666667')

    start6 = datetime.date(2020, 1, 30)
    asof6 = datetime.date(2020, 3, 31)
    assert round(dcfc_30_e_360(start6, asof6, asof6), 14) == Decimal('0.16666666666667')

    start7 = datetime.date(2020, 1, 30)
    asof7 = datetime.date(2020, 3, 30)
    assert round(dcfc_30_e_360(start7, asof7, asof7), 14) == Decimal('0.16666666666667')

    start8 = datetime.date(2020, 1, 31)
    asof8 = datetime.date(2020, 3, 30)
    assert round(dcfc_30_e_360(start8, asof8, asof8), 14) == Decimal('0.16666666666667')


# LLM-generated content at query #18
#--------------------------

```python
def test_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #19
#--------------------------

```python
def test_dcfc_act_365_l_with_leap_year():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_365_l(start, asof, asof)
    assert round(result, 14) == Decimal('0.17213114754098')

def test_dcfc_act_365_l_with_non_leap_year():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_365_l(start, asof, asof)
    assert round(result, 14) == Decimal('0.16939890710383')

def test_dcfc_act_365_l_with_multiple_years():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act_365_l(start, asof, asof)
    assert round(result, 14) == Decimal('1.08196721311475')

def test_dcfc_act_365_l_with_long_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_365_l(start, asof, asof)
    assert round(result, 14) == Decimal('1.32876712328767')


# LLM-generated content at query #20
#--------------------------

Here are the test cases for the `calculate_daily_fraction` method:


# LLM-generated content at query #21
#--------------------------

```python
def test_register_successfully_registers_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TestAlt"}, currencies={Currencies["USD"]}, calculate_fraction_method=lambda x, y, z, w: Decimal(1))
    registry.register(dcc)
    assert registry._buffer_main["TestDCC"] == dcc
    assert registry._buffer_altn["TestAlt"] == dcc

def test_register_raises_error_when_dcc_is_already_registered():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TestAlt"}, currencies={Currencies["USD"]}, calculate_fraction_method=lambda x, y, z, w: Decimal(1))
    registry.register(dcc)
    try:
        registry.register(dcc)
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_register_raises_error_when_altname_is_already_registered():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC1", altnames={"TestAlt"}, currencies={Currencies["USD"]}, calculate_fraction_method=lambda x, y, z, w: Decimal(1))
    dcc2 = DCC(name="TestDCC2", altnames={"TestAlt"}, currencies={Currencies["USD"]}, calculate_fraction_method=lambda x, y, z, w: Decimal(1))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Expected TypeError"
    except TypeError:
        pass


# LLM-generated content at query #22
#--------------------------

```python
def test_last_payment_date_returns_start_date_when_invalid():
    start_date = datetime.date(2014, 1, 1)
    asof_date = datetime.date(2015, 12, 31)
    frequency = 1
    eom = 0
    result = _last_payment_date(start_date, asof_date, frequency, eom)
    assert result == start_date


# LLM-generated content at query #23
#--------------------------

```python
def test_dcfc_nl_365_with_leap_day():
    start = datetime.date(2008, 2, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    expected = Decimal('0.00273972602740')
    assert round(result, 14) == round(expected, 14)

def test_dcfc_nl_365_without_leap_day():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    expected = Decimal('0.16986301369863')
    assert round(result, 14) == round(expected, 14)

def test_dcfc_nl_365_same_day():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2007, 12, 28)
    end = datetime.date(2007, 12, 28)
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    assert result == Decimal('0')

def test_dcfc_nl_365_multi_year_period():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    expected = Decimal('1.08219178082192')
    assert round(result, 14) == round(expected, 14)

def test_dcfc_nl_365_long_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    expected = Decimal('1.32602739726027')
    assert round(result, 14) == round(expected, 14)


# LLM-generated content at query #24
#--------------------------

```python
def test_dcfc_30_360_isda_predicate_evaluates_to_false():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_360_isda(start, asof, end)
    assert result != ZERO


# LLM-generated content at query #25
#--------------------------

```python
def test_is_last_day_of_month_last_day():
    date = Date(2023, 1, 31)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_not_last_day():
    date = Date(2023, 1, 30)
    assert _is_last_day_of_month(date) == False

def test_is_last_day_of_month_february_non_leap_year():
    date = Date(2023, 2, 28)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_february_leap_year():
    date = Date(2024, 2, 29)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_february_not_leap_year():
    date = Date(2023, 2, 29)
    assert _is_last_day_of_month(date) == False


# LLM-generated content at query #26
#--------------------------

```python
def test_calculate_daily_fraction_asof_minus_1_less_than_start():
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 1, 2)
    end = datetime.date(2023, 1, 3)
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda *args: Decimal(1))
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(1)


# LLM-generated content at query #27
#--------------------------

```python
def test_dcfc_30_e_360_does_not_modify_asof_when_not_31st():
    start = datetime.date(2023, 1, 15)
    asof = datetime.date(2023, 2, 28)  # Not 31st
    end = datetime.date(2023, 2, 28)
    original_asof = asof
    
    dcfc_30_e_360(start, asof, end)
    
    assert asof == original_asof


# LLM-generated content at query #28
#--------------------------

```python
def test_coupon_calculation():
    dcc = DCC(
        name="30/360",
        altnames={"Bond Basis"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda start, asof, end, freq: Decimal((asof - start).days) / Decimal(360)
    )
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 6, 30)
    end = datetime.date(2023, 12, 31)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(Decimal("25.00"), Currency("USD"))


# LLM-generated content at query #29
#--------------------------

```python
def test_is_registered_returns_true_when_name_is_in_main_or_alt_buffer():
    dcc_registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"AltDCC"}, currencies=set(), calculate_fraction_method=lambda *args: Decimal("0.0"))
    dcc_registry.register(dcc)
    assert dcc_registry._is_registered("TestDCC")
    assert dcc_registry._is_registered("AltDCC")


# LLM-generated content at query #30
#--------------------------

```
def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #31
#--------------------------

```python
def test_dcfc_30_e_plus_360_asof_day_31():
    start = datetime.date(2023, 10, 31)
    asof = datetime.date(2023, 11, 30)
    end = datetime.date(2023, 11, 30)
    result = dcfc_30_e_plus_360(start, asof, end)
    assert result == Decimal("0.08333333333333")


# LLM-generated content at query #32
#--------------------------

```python
def test_dcfc_act_365_l_with_leap_year():
    start = datetime.date(2008, 1, 1)
    asof = datetime.date(2008, 2, 29)  # Leap day
    end = datetime.date(2008, 2, 29)
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    assert result == Decimal('0.16939890710383')  # (29 + 31) / 366

def test_dcfc_act_365_l_with_non_leap_year():
    start = datetime.date(2007, 1, 1)
    asof = datetime.date(2007, 2, 28)  # Non-leap year
    end = datetime.date(2007, 2, 28)
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    assert result == Decimal('0.15890410958904')  # (28 + 31) / 365

def test_dcfc_act_365_l_with_year_boundary():
    start = datetime.date(2007, 12, 31)
    asof = datetime.date(2008, 1, 31)  # Crosses into leap year
    end = datetime.date(2008, 1, 31)
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    assert result == Decimal('0.08688524590164')  # 31 / 366

def test_dcfc_act_365_l_with_single_day_leap_year():
    start = datetime.date(2008, 2, 29)
    asof = datetime.date(2008, 2, 29)  # Single day in leap year
    end = datetime.date(2008, 2, 29)
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    assert result == Decimal('0.00273224043716')  # 1 / 366


# LLM-generated content at query #33
#--------------------------

```python
def test_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #34
#--------------------------

```python
def test_coupon_calculation():
    dcc = DCC(
        name="Actual/Actual",
        altnames={"Act/Act"},
        currencies={USD, EUR},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal((e - s).days)
    )
    principal = Money(1000, USD)
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 1)
    freq = Decimal(2)
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(25, USD)

def test_coupon_with_eom():
    dcc = DCC(
        name="Actual/Actual",
        altnames={"Act/Act"},
        currencies={USD, EUR},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal((e - s).days)
    )
    principal = Money(1000, USD)
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 31)
    freq = Decimal(2)
    eom = 31
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(25, USD)

def test_coupon_before_start_date():
    dcc = DCC(
        name="Actual/Actual",
        altnames={"Act/Act"},
        currencies={USD, EUR},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal((e - s).days)
    )
    principal = Money(1000, USD)
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2019, 12, 31)
    end = datetime.date(2021, 1, 1)
    freq = Decimal(2)
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(0, USD)

def test_coupon_after_end_date():
    dcc = DCC(
        name="Actual/Actual",
        altnames={"Act/Act"},
        currencies={USD, EUR},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal((e - s).days)
    )
    principal = Money(1000, USD)
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 2)
    end = datetime.date(2021, 1, 1)
    freq = Decimal(2)
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(0, USD)

def test_coupon_on_payment_date():
    dcc = DCC(
        name="Actual/Actual",
        altnames={"Act/Act"},
        currencies={USD, EUR},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal((e - s).days)
    )
    principal = Money(1000, USD)
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2021, 1, 1)
    freq = Decimal(2)
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(25, USD)


# LLM-generated content at query #35
#--------------------------

def test_dcfc_act_act_with_non_leap_year():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 3, 1)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('0.16164383561643836')
    assert result == expected

def test_dcfc_act_act_with_leap_year():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 3, 1)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('0.16393442622950818')
    assert result == expected

def test_dcfc_act_act_spanning_leap_and_non_leap_years():
    start = datetime.date(2019, 12, 1)
    asof = datetime.date(2020, 2, 1)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('0.16939890710382514')
    assert result == expected

def test_dcfc_act_act_with_single_day():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 1, 2)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('0.0027397260273972603')
    assert result == expected

def test_dcfc_act_act_with_full_year():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2022, 1, 1)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('1.0')
    assert result == expected

def test_dcfc_act_act_with_full_leap_year():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('1.0')
    assert result == expected


