####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_last_day_of_month_true():
    date = Date(2023, 1, 31)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_false():
    date = Date(2023, 1, 30)
    assert _is_last_day_of_month(date) == False

def test_is_last_day_of_month_feb_non_leap():
    date = Date(2023, 2, 28)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_feb_leap():
    date = Date(2024, 2, 29)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_april():
    date = Date(2023, 4, 30)
    assert _is_last_day_of_month(date) == True


# LLM-generated content at query #2
#--------------------------

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

def test_dcfc_30_360_isda_same_date_case():
    date = datetime.date(2020, 1, 1)
    result = dcfc_30_360_isda(date, date, date)
    assert result == Decimal(0)

def test_dcfc_30_360_isda_end_of_month_adjustment():
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 28)
    end = datetime.date(2020, 2, 28)
    result = dcfc_30_360_isda(start, asof, end)
    assert round(result, 14) == Decimal('0.08333333333333')


# LLM-generated content at query #3
#--------------------------

```python
def test_register_new_dcc():
    dcc = DCC(
        name="TestDCC",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"),
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
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"),
    )
    dcc2 = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.2"),
    )
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Should raise TypeError"
    except TypeError:
        pass

def test_register_duplicate_alt_name():
    dcc1 = DCC(
        name="TestDCC1",
        altnames={"TestAlt"},
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"),
    )
    dcc2 = DCC(
        name="TestDCC2",
        altnames={"TestAlt"},
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.2"),
    )
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Should raise TypeError"
    except TypeError:
        pass

def test_register_duplicate_name_in_alt_names():
    dcc1 = DCC(
        name="TestDCC1",
        altnames={"TestAlt"},
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"),
    )
    dcc2 = DCC(
        name="TestAlt",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.2"),
    )
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False, "Should raise TypeError"
    except TypeError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_next_payment_date_basic():
    start = datetime.date(2014, 1, 1)
    result = _next_payment_date(start, 1)
    assert result == datetime.date(2015, 1, 1)

def test_next_payment_date_with_eom():
    start = datetime.date(2014, 1, 1)
    result = _next_payment_date(start, 1, 15)
    assert result == datetime.date(2015, 1, 15)

def test_next_payment_date_monthly_frequency():
    start = datetime.date(2023, 3, 15)
    result = _next_payment_date(start, 12)
    assert result == datetime.date(2023, 4, 15)

def test_next_payment_date_eom_february():
    start = datetime.date(2023, 1, 31)
    result = _next_payment_date(start, 1, 31)
    assert result == datetime.date(2023, 2, 28)

def test_next_payment_date_quarterly_frequency():
    start = datetime.date(2023, 1, 1)
    result = _next_payment_date(start, 4)
    assert result == datetime.date(2023, 4, 1)

def test_next_payment_date_semi_annual_frequency():
    start = datetime.date(2023, 1, 1)
    result = _next_payment_date(start, 2)
    assert result == datetime.date(2023, 7, 1)


# LLM-generated content at query #5
#--------------------------

```python
def test_register_new_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda x, y, z, w: Decimal(0))
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("TestAlt") == dcc

def test_register_existing_dcc():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda x, y, z, w: Decimal(0))
    registry.register(dcc1)
    dcc2 = DCC(name="TestDCC", altnames={"TestAlt2"}, currencies={}, calculate_fraction_method=lambda x, y, z, w: Decimal(0))
    try:
        registry.register(dcc2)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"

def test_register_conflicting_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC1", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda x, y, z, w: Decimal(0))
    registry.register(dcc1)
    dcc2 = DCC(name="TestDCC2", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda x, y, z, w: Decimal(0))
    try:
        registry.register(dcc2)
    except TypeError:
        pass
    else:
        assert False, "Expected TypeError"


# LLM-generated content at query #6
#--------------------------

```python
def test_dcfc_30_360_us_with_valid_dates():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_us(start, asof, asof)
    assert result == Decimal('0.16666666666667')

def test_dcfc_30_360_us_with_leap_year():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_us(start, asof, asof)
    assert result == Decimal('0.16944444444444')

def test_dcfc_30_360_us_with_month_end_dates():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_us(start, asof, asof)
    assert result == Decimal('1.08333333333333')

def test_dcfc_30_360_us_with_long_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_us(start, asof, asof)
    assert result == Decimal('1.33333333333333')


# LLM-generated content at query #7
#--------------------------

```
def test_register_new_dcc():
    dcc = DCC(
        name="TestDCC",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
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
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    dcc2 = DCC(
        name="TestDCC",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
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
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    dcc2 = DCC(
        name="TestDCC2",
        altnames={"TestAlt"},
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError:
        assert True

def test_register_duplicate_alt_name_in_main():
    dcc1 = DCC(
        name="TestDCC1",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    dcc2 = DCC(
        name="TestAlt",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError:
        assert True


# LLM-generated content at query #8
#--------------------------

```python
def test_dcfc_30_e_360_example1():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('0.16666666666667')

def test_dcfc_30_e_360_example2():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('0.16944444444444')

def test_dcfc_30_e_360_example3():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('1.08333333333333')

def test_dcfc_30_e_360_example4():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start, asof, asof)
    assert round(result, 14) == Decimal('1.33055555555556')


# LLM-generated content at query #9
#--------------------------

def test_dcfc_nl_365_with_same_day():
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 1)
    end = datetime.date(2017, 1, 1)
    assert dcfc_nl_365(start, asof, end) == Decimal('0')

def test_dcfc_nl_365_with_one_day():
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 2)
    end = datetime.date(2017, 1, 2)
    assert dcfc_nl_365(start, asof, end) == Decimal('0.002739726027397260273972602739726027397260274')

def test_dcfc_nl_365_with_leap_year_in_range():
    start = datetime.date(2016, 1, 1)
    asof = datetime.date(2016, 12, 31)
    end = datetime.date(2016, 12, 31)
    assert round(dcfc_nl_365(start, asof, end), 14) == Decimal('0.99726027397260')

def test_dcfc_nl_365_with_leap_day_in_range():
    start = datetime.date(2016, 2, 28)
    asof = datetime.date(2016, 3, 1)
    end = datetime.date(2016, 3, 1)
    assert round(dcfc_nl_365(start, asof, end), 14) == Decimal('0.00821917808219')

def test_dcfc_nl_365_with_multiple_years():
    start = datetime.date(2015, 1, 1)
    asof = datetime.date(2017, 1, 1)
    end = datetime.date(2017, 1, 1)
    assert round(dcfc_nl_365(start, asof, end), 14) == Decimal('2.00273972602740')


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_is_last_day_of_month_true():
    date = Date(2023, 1, 31)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_false():
    date = Date(2023, 1, 15)
    result = _is_last_day_of_month(date)
    assert result == False

def test_is_last_day_of_month_february_non_leap():
    date = Date(2023, 2, 28)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_february_leap():
    date = Date(2024, 2, 29)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_december():
    date = Date(2023, 12, 31)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_april():
    date = Date(2023, 4, 30)
    result = _is_last_day_of_month(date)
    assert result == True


# LLM-generated content at query #2
#--------------------------

def test_get_date_range_with_single_day():
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 2)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 1, 1)]
    assert result == expected

def test_get_date_range_with_multiple_days():
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 4)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2), datetime.date(2023, 1, 3)]
    assert result == expected

def test_get_date_range_with_zero_days():
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 1)
    result = list(_get_date_range(start, end))
    expected = []
    assert result == expected

def test_get_date_range_with_negative_days():
    start = datetime.date(2023, 1, 5)
    end = datetime.date(2023, 1, 1)
    result = list(_get_date_range(start, end))
    expected = []
    assert result == expected

def test_get_date_range_with_leap_year():
    start = datetime.date(2024, 2, 28)
    end = datetime.date(2024, 3, 2)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2024, 2, 28), datetime.date(2024, 2, 29), datetime.date(2024, 3, 1)]
    assert result == expected

def test_get_date_range_with_month_crossing():
    start = datetime.date(2023, 1, 30)
    end = datetime.date(2023, 2, 3)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 1, 30), datetime.date(2023, 1, 31), datetime.date(2023, 2, 1), datetime.date(2023, 2, 2)]
    assert result == expected

def test_get_date_range_with_year_crossing():
    start = datetime.date(2023, 12, 30)
    end = datetime.date(2024, 1, 2)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 12, 30), datetime.date(2023, 12, 31), datetime.date(2024, 1, 1)]
    assert result == expected


# LLM-generated content at query #3
#--------------------------

def test_get_date_range_with_single_day():
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 2)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 1, 1)]
    assert result == expected

def test_get_date_range_with_multiple_days():
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 4)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2), datetime.date(2023, 1, 3)]
    assert result == expected

def test_get_date_range_with_zero_days():
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 1)
    result = list(_get_date_range(start, end))
    expected = []
    assert result == expected

def test_get_date_range_across_month():
    start = datetime.date(2023, 1, 30)
    end = datetime.date(2023, 2, 2)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 1, 30), datetime.date(2023, 1, 31), datetime.date(2023, 2, 1)]
    assert result == expected

def test_get_date_range_with_leap_year():
    start = datetime.date(2024, 2, 28)
    end = datetime.date(2024, 3, 2)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2024, 2, 28), datetime.date(2024, 2, 29), datetime.date(2024, 3, 1)]
    assert result == expected


# LLM-generated content at query #4
#--------------------------

def test_register_successfully_adds_new_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TDCC"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("TDCC") == dcc

def test_register_raises_error_on_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="TestDCC", altnames={"Other"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'TestDCC' is already registered"

def test_register_raises_error_on_duplicate_altname_in_main_buffer():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="FirstDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="SecondDCC", altnames={"FirstDCC"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'SecondDCC' is already registered"

def test_register_raises_error_on_duplicate_altname_in_alt_buffer():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="FirstDCC", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="SecondDCC", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'SecondDCC' is already registered"

def test_register_raises_error_when_main_name_exists_as_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="FirstDCC", altnames={"SecondDCC"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="SecondDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'SecondDCC' is already registered"

def test_register_adds_all_altnames_to_buffer():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"Alt1", "Alt2", "Alt3"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("Alt1") == dcc
    assert registry._find_strict("Alt2") == dcc
    assert registry._find_strict("Alt3") == dcc

def test_register_main_buffer_contains_dcc_by_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TDCC"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._buffer_main["TestDCC"] == dcc
    assert "TestDCC" not in registry._buffer_altn

def test_register_alt_buffer_contains_dcc_by_altnames():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TDCC", "Test"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._buffer_altn["TDCC"] == dcc
    assert registry._buffer_altn["Test"] == dcc
    assert "TDCC" not in registry._buffer_main
    assert "Test" not in registry._buffer_main

def test_register_empty_altnames_does_not_add_to_alt_buffer():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert len(registry._buffer_altn) == 0
    assert registry._buffer_main["TestDCC"] == dcc


# LLM-generated content at query #5
#--------------------------

def test_register_successfully_adds_new_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TD"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("TD") == dcc

def test_register_raises_error_if_main_name_already_registered():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc1)
    dcc2 = DCC(name="TestDCC", altnames={"Other"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'TestDCC' is already registered"

def test_register_raises_error_if_altname_conflicts_with_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc1)
    dcc2 = DCC(name="AnotherDCC", altnames={"TestDCC"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'AnotherDCC' is already registered"

def test_register_raises_error_if_altname_conflicts_with_existing_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC", altnames={"Alt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc1)
    dcc2 = DCC(name="AnotherDCC", altnames={"Alt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'AnotherDCC' is already registered"

def test_register_adds_all_altnames_to_buffer():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"Alt1", "Alt2"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("Alt1") == dcc
    assert registry._find_strict("Alt2") == dcc

def test_register_does_not_modify_existing_registries():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="First", altnames={"F1"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"))
    registry.register(dcc1)
    dcc2 = DCC(name="Second", altnames={"S1"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.2"))
    registry.register(dcc2)
    assert registry._find_strict("First") == dcc1
    assert registry._find_strict("F1") == dcc1
    assert registry._find_strict("Second") == dcc2
    assert registry._find_strict("S1") == dcc2


# LLM-generated content at query #6
#--------------------------

def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #7
#--------------------------

def test_dcfc_30_360_us_basic():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_leap_year():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_last_day_of_month_start():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_last_day_of_month_both():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('1.33333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_same_date():
    start = datetime.date(2023, 1, 15)
    asof = start
    end = start
    result = dcfc_30_360_us(start, asof, end)
    assert result == Decimal('0')

def test_dcfc_30_360_us_d1_31_d2_31():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('0.07777777777778')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_d1_30_d2_31():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('0.07777777777778')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_d1_31_d2_30():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('0.07777777777778')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_cross_year():
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 1, 15)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('0.08333333333333')
    assert round(result, 14) == expected


# LLM-generated content at query #8
#--------------------------

def test_register_raises_type_error_when_altname_conflict():
    dcc1 = DCC(name="DCC1", altnames={"ALT1"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0"))
    dcc2 = DCC(name="DCC2", altnames={"ALT1"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0"))
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'DCC2' is already registered"


# LLM-generated content at query #9
#--------------------------

def test_is_last_day_of_month_true():
    date = Date(2023, 1, 31)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_false():
    date = Date(2023, 1, 15)
    result = _is_last_day_of_month(date)
    assert result == False

def test_is_last_day_of_month_february_non_leap():
    date = Date(2023, 2, 28)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_february_leap():
    date = Date(2024, 2, 29)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_december():
    date = Date(2023, 12, 31)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_april():
    date = Date(2023, 4, 30)
    result = _is_last_day_of_month(date)
    assert result == True


# LLM-generated content at query #10
#--------------------------

def test_dcfc_30_360_isda_example1():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

def test_dcfc_30_360_isda_example2():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

def test_dcfc_30_360_isda_example3():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_360_isda_example4():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('1.33333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_360_isda_start_day_31_adjustment():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    nod = (28 - 30) + 30 * (2 - 1) + 360 * (2023 - 2023)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_isda_start_day_30_asof_day_31_adjustment():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    end = datetime.date(2023, 2, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    nod = (30 - 30) + 30 * (2 - 1) + 360 * (2023 - 2023)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_isda_no_adjustment():
    start = datetime.date(2023, 3, 15)
    asof = datetime.date(2023, 6, 15)
    end = datetime.date(2023, 6, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    nod = (15 - 15) + 30 * (6 - 3) + 360 * (2023 - 2023)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_isda_cross_year():
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 3, 15)
    end = datetime.date(2023, 3, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    nod = (15 - 15) + 30 * (3 - 12) + 360 * (2023 - 2022)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_isda_leap_year_feb29():
    start = datetime.date(2020, 2, 29)
    asof = datetime.date(2020, 3, 31)
    end = datetime.date(2020, 3, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    nod = (30 - 29) + 30 * (3 - 2) + 360 * (2020 - 2020)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_isda_same_date():
    start = datetime.date(2023, 5, 10)
    asof = datetime.date(2023, 5, 10)
    end = datetime.date(2023, 5, 10)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal(0) / Decimal(360)
    assert result == expected


# LLM-generated content at query #11
#--------------------------

def test_dcfc_30_e_360_example1():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_example2():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_example3():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_example4():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('1.33055555555556')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_start_day_31_adjustment():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_e_360(start, asof, end)
    nod = (28 - 30) + 30 * (2 - 1) + 360 * (2023 - 2023)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_asof_day_31_adjustment():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 3, 31)
    end = datetime.date(2023, 3, 31)
    result = dcfc_30_e_360(start, asof, end)
    nod = (30 - 30) + 30 * (3 - 1) + 360 * (2023 - 2023)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_both_days_31_adjustment():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    end = datetime.date(2023, 3, 31)
    result = dcfc_30_e_360(start, asof, end)
    nod = (30 - 30) + 30 * (3 - 1) + 360 * (2023 - 2023)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_no_adjustment():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_e_360(start, asof, end)
    nod = (28 - 30) + 30 * (2 - 1) + 360 * (2023 - 2023)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_cross_year():
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 1, 15)
    end = datetime.date(2023, 1, 15)
    result = dcfc_30_e_360(start, asof, end)
    nod = (15 - 15) + 30 * (1 - 12) + 360 * (2023 - 2022)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_same_date():
    start = datetime.date(2023, 5, 15)
    asof = datetime.date(2023, 5, 15)
    end = datetime.date(2023, 5, 15)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal(0)
    assert result == expected


# LLM-generated content at query #12
#--------------------------

def test_dcfc_nl_365_basic_calculation():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('0.16986301369863')
    assert round(result, 14) == expected

def test_dcfc_nl_365_with_leap_day_in_range():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('0.16986301369863')
    assert round(result, 14) == expected

def test_dcfc_nl_365_long_period():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('1.08219178082192')
    assert round(result, 14) == expected

def test_dcfc_nl_365_another_long_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('1.32602739726027')
    assert round(result, 14) == expected

def test_dcfc_nl_365_same_start_and_asof():
    start = datetime.date(2020, 1, 1)
    asof = start
    end = asof
    result = dcfc_nl_365(start, asof, end)
    assert result == Decimal('0')

def test_dcfc_nl_365_no_leap_day_in_range():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 12, 31)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('364') / Decimal('365')
    assert result == expected

def test_dcfc_nl_365_with_leap_day_excluded():
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('2') / Decimal('365')
    assert result == expected

def test_dcfc_nl_365_with_leap_day_included():
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 2, 29)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('1') / Decimal('365')
    assert result == expected

def test_dcfc_nl_365_crossing_multiple_leap_years():
    start = datetime.date(2016, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    total_days = (asof - start).days
    leap_days = 2
    expected = Decimal(total_days - leap_days) / Decimal('365')
    assert result == expected

def test_dcfc_nl_365_freq_parameter_ignored():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    freq = Decimal('2')
    result = dcfc_nl_365(start, asof, end, freq)
    expected = Decimal('0.16986301369863')
    assert round(result, 14) == expected


# LLM-generated content at query #13
#--------------------------

def test_last_payment_date_annual_frequency():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_same_year_annual():
    result = _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_semiannual_frequency():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2)
    expected = datetime.date(2015, 7, 1)
    assert result == expected

def test_last_payment_date_semiannual_before_midyear():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2)
    expected = datetime.date(2015, 7, 1)
    assert result == expected

def test_last_payment_date_semiannual_early_year():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_annual_start_midyear():
    result = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1)
    expected = datetime.date(2014, 6, 1)
    assert result == expected

def test_last_payment_date_quarterly_frequency():
    result = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4)
    expected = datetime.date(2015, 7, 7)
    assert result == expected

def test_last_payment_date_annual_near_year_end():
    result = _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1)
    expected = datetime.date(2014, 12, 9)
    assert result == expected

def test_last_payment_date_semiannual_december_start():
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2)
    expected = datetime.date(2015, 12, 15)
    assert result == expected

def test_last_payment_date_semiannual_end_of_year():
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    expected = datetime.date(2015, 12, 15)
    assert result == expected

def test_last_payment_date_eom_handling():
    result = _last_payment_date(datetime.date(2014, 1, 31), datetime.date(2015, 3, 15), 1, eom=31)
    expected = datetime.date(2014, 1, 31)
    assert result == expected

def test_last_payment_date_month_end_adjustment():
    result = _last_payment_date(datetime.date(2014, 2, 28), datetime.date(2015, 3, 31), 1, eom=31)
    expected = datetime.date(2014, 2, 28)
    assert result == expected

def test_last_payment_date_frequency_decimal():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), Decimal(1))
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_start_date_returned():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2014, 1, 1), 1)
    expected = datetime.date(2014, 1, 1)
    assert result == expected

def test_last_payment_date_before_first_payment():
    result = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2014, 5, 31), 1)
    expected = datetime.date(2014, 6, 1)
    assert result == expected


# LLM-generated content at query #14
#--------------------------

def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #15
#--------------------------

def test_dcfc_act_365_l_basic_calculation():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    expected = Decimal('0.16939890710383')
    assert round(result, 14) == expected

def test_dcfc_act_365_l_leap_year():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    expected = Decimal('0.17213114754098')
    assert round(result, 14) == expected

def test_dcfc_act_365_l_long_period():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    expected = Decimal('1.08196721311475')
    assert round(result, 14) == expected

def test_dcfc_act_365_l_another_long_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    expected = Decimal('1.32876712328767')
    assert round(result, 14) == expected

def test_dcfc_act_365_l_same_date():
    start = datetime.date(2020, 1, 1)
    asof = start
    end = start
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    assert result == Decimal('0')

def test_dcfc_act_365_l_non_leap_year_denominator():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 12, 31)
    end = asof
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    expected = Decimal('364') / Decimal('365')
    assert result == expected

def test_dcfc_act_365_l_leap_year_denominator():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = asof
    result = dcfc_act_365_l(start=start, asof=asof, end=end)
    expected = Decimal('365') / Decimal('366')
    assert result == expected


# LLM-generated content at query #16
#--------------------------

def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #17
#--------------------------

def test_next_payment_date_no_eom():
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, None)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_next_payment_date_with_eom():
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, 15)
    expected = datetime.date(2015, 1, 15)
    assert result == expected

def test_next_payment_date_frequency_two():
    result = _next_payment_date(datetime.date(2014, 1, 1), 2, None)
    expected = datetime.date(2014, 7, 1)
    assert result == expected

def test_next_payment_date_frequency_two_with_eom():
    result = _next_payment_date(datetime.date(2014, 1, 1), 2, 31)
    expected = datetime.date(2014, 7, 31)
    assert result == expected

def test_next_payment_date_frequency_four():
    result = _next_payment_date(datetime.date(2014, 1, 1), 4, None)
    expected = datetime.date(2014, 4, 1)
    assert result == expected

def test_next_payment_date_frequency_four_with_eom():
    result = _next_payment_date(datetime.date(2014, 1, 1), 4, 30)
    expected = datetime.date(2014, 4, 30)
    assert result == expected

def test_next_payment_date_invalid_eom():
    result = _next_payment_date(datetime.date(2014, 2, 1), 1, 31)
    expected = datetime.date(2015, 2, 1)
    assert result == expected

def test_next_payment_date_decimal_frequency():
    result = _next_payment_date(datetime.date(2014, 1, 1), Decimal('0.5'), None)
    expected = datetime.date(2026, 1, 1)
    assert result == expected

def test_next_payment_date_decimal_frequency_with_eom():
    result = _next_payment_date(datetime.date(2014, 1, 1), Decimal('0.5'), 10)
    expected = datetime.date(2026, 1, 10)
    assert result == expected

def test_next_payment_date_frequency_six():
    result = _next_payment_date(datetime.date(2014, 1, 1), 6, None)
    expected = datetime.date(2014, 3, 1)
    assert result == expected

def test_next_payment_date_frequency_six_with_eom():
    result = _next_payment_date(datetime.date(2014, 1, 1), 6, 28)
    expected = datetime.date(2014, 3, 28)
    assert result == expected

def test_next_payment_date_leap_year():
    result = _next_payment_date(datetime.date(2020, 2, 29), 1, None)
    expected = datetime.date(2021, 2, 28)
    assert result == expected

def test_next_payment_date_leap_year_with_eom():
    result = _next_payment_date(datetime.date(2020, 2, 29), 1, 29)
    expected = datetime.date(2021, 2, 28)
    assert result == expected


# LLM-generated content at query #18
#--------------------------

def test_last_payment_date_annual_frequency():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_same_year_annual():
    start = datetime.date(2015, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_semi_annual_frequency():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 7, 1)
    assert result == expected

def test_last_payment_date_semi_annual_before_mid_year():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 8, 31)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 7, 1)
    assert result == expected

def test_last_payment_date_semi_annual_early_year():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 4, 30)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_annual_start_mid_year():
    start = datetime.date(2014, 6, 1)
    asof = datetime.date(2015, 4, 30)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2014, 6, 1)
    assert result == expected

def test_last_payment_date_quarterly_frequency():
    start = datetime.date(2008, 7, 7)
    asof = datetime.date(2015, 10, 6)
    frequency = 4
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 7, 7)
    assert result == expected

def test_last_payment_date_annual_december_start():
    start = datetime.date(2014, 12, 9)
    asof = datetime.date(2015, 12, 4)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2014, 12, 9)
    assert result == expected

def test_last_payment_date_semi_annual_december_start():
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2016, 1, 6)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 12, 15)
    assert result == expected

def test_last_payment_date_semi_annual_december_end_year():
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2015, 12, 31)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 12, 15)
    assert result == expected

def test_last_payment_date_with_eom_override():
    start = datetime.date(2014, 1, 31)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = 15
    result = _last_payment_date(start, asof, frequency, eom)
    expected = datetime.date(2015, 1, 15)
    assert result == expected

def test_last_payment_date_frequency_decimal():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = Decimal('1')
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_edge_case_negative_year():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(1, 1, 1)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = start
    assert result == expected

def test_last_payment_date_monthly_frequency():
    start = datetime.date(2014, 1, 15)
    asof = datetime.date(2015, 12, 20)
    frequency = 12
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 12, 15)
    assert result == expected

def test_last_payment_date_bi_monthly_frequency():
    start = datetime.date(2014, 1, 15)
    asof = datetime.date(2015, 12, 20)
    frequency = 6
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 11, 15)
    assert result == expected


# LLM-generated content at query #19
#--------------------------

def test_coupon_basic_annual():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2021, 1, 1)
    freq = 1
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(Decimal("25"), "USD")
    assert result == expected

def test_coupon_semi_annual():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.25"))
    principal = Money(Decimal("2000"), "EUR")
    rate = Decimal("0.03")
    start = datetime.date(2019, 6, 15)
    asof = datetime.date(2019, 9, 15)
    end = datetime.date(2020, 6, 15)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(Decimal("15"), "EUR")
    assert result == expected

def test_coupon_quarterly():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"))
    principal = Money(Decimal("5000"), "GBP")
    rate = Decimal("0.04")
    start = datetime.date(2021, 3, 10)
    asof = datetime.date(2021, 4, 10)
    end = datetime.date(2022, 3, 10)
    freq = 4
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(Decimal("20"), "GBP")
    assert result == expected

def test_coupon_with_eom():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    principal = Money(Decimal("1500"), "JPY")
    rate = Decimal("0.02")
    start = datetime.date(2018, 2, 28)
    asof = datetime.date(2018, 5, 31)
    end = datetime.date(2019, 2, 28)
    freq = 1
    eom = 31
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = Money(Decimal("9"), "JPY")
    assert result == expected

def test_coupon_asof_on_start():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.0"))
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = 1
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(Decimal("0"), "USD")
    assert result == expected

def test_coupon_asof_on_end():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("1.0"))
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = 1
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(Decimal("50"), "USD")
    assert result == expected

def test_coupon_fraction_zero():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.0"))
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(Decimal("0"), "USD")
    assert result == expected

def test_coupon_fraction_full_period():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("1.0"))
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = 1
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(Decimal("50"), "USD")
    assert result == expected

def test_coupon_with_high_frequency():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.08333"))
    principal = Money(Decimal("12000"), "CAD")
    rate = Decimal("0.06")
    start = datetime.date(2022, 1, 1)
    asof = datetime.date(2022, 2, 1)
    end = datetime.date(2023, 1, 1)
    freq = 12
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(Decimal("60"), "CAD")
    assert result == expected

def test_coupon_negative_rate():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("1000"), "USD")
    rate = Decimal("-0.02")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2021, 1, 1)
    freq = 1
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(Decimal("-10"), "USD")
    assert result == expected


# LLM-generated content at query #20
#--------------------------

def test_dcfc_act_act_basic():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('0.16942884946478')
    assert round(result, 14) == expected

def test_dcfc_act_act_leap_year():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('0.17216108990194')
    assert round(result, 14) == expected

def test_dcfc_act_act_multi_year():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('1.08243131970956')
    assert round(result, 14) == expected

def test_dcfc_act_act_long_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('1.32625945055768')
    assert round(result, 14) == expected

def test_dcfc_act_act_same_day():
    start = datetime.date(2020, 1, 1)
    asof = start
    end = asof
    result = dcfc_act_act(start, asof, end)
    assert result == Decimal('0')

def test_dcfc_act_act_one_day():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('1') / Decimal(366)
    assert result == expected

def test_dcfc_act_act_non_leap_year():
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 1, 2)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('1') / Decimal(365)
    assert result == expected

def test_dcfc_act_act_cross_year_boundary():
    start = datetime.date(2019, 12, 31)
    asof = datetime.date(2020, 1, 1)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('1') / Decimal(365)
    assert result == expected

def test_dcfc_act_act_full_leap_year():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    end = asof
    result = dcfc_act_act(start, asof, end)
    assert result == Decimal('1')

def test_dcfc_act_act_full_non_leap_year():
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = asof
    result = dcfc_act_act(start, asof, end)
    assert result == Decimal('1')


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_dcfc_nl_365_standard_period():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('0.16986301369863')
    assert round(result, 14) == expected

def test_dcfc_nl_365_leap_day_in_period():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('0.16986301369863')
    assert round(result, 14) == expected

def test_dcfc_nl_365_longer_period():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('1.08219178082192')
    assert round(result, 14) == expected

def test_dcfc_nl_365_another_long_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('1.32602739726027')
    assert round(result, 14) == expected

def test_dcfc_nl_365_same_start_and_asof():
    start = datetime.date(2020, 1, 1)
    asof = start
    end = asof
    result = dcfc_nl_365(start, asof, end)
    assert result == Decimal('0')

def test_dcfc_nl_365_one_day_period():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('1') / Decimal('365')
    assert result == expected

def test_dcfc_nl_365_period_with_leap_day_excluded():
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('2') / Decimal('365')
    assert result == expected

def test_dcfc_nl_365_period_including_leap_day():
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('2') / Decimal('365')
    assert result == expected

def test_dcfc_nl_365_period_spanning_leap_year():
    start = datetime.date(2019, 12, 31)
    asof = datetime.date(2020, 12, 31)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    expected = Decimal('365') / Decimal('365')
    assert result == expected

def test_dcfc_nl_365_asof_before_start():
    start = datetime.date(2020, 1, 2)
    asof = datetime.date(2020, 1, 1)
    end = asof
    result = dcfc_nl_365(start, asof, end)
    assert result == Decimal('0')


# LLM-generated content at query #2
#--------------------------

def test_interest_calculates_correctly():
    dcc = DCC(name="Actual/360", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 30)
    end = Date(2023, 12, 31)
    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money(Decimal("25"), Currency("USD"))
    assert result == expected

def test_interest_without_end_uses_asof():
    dcc = DCC(name="Actual/360", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.25"))
    principal = Money(Decimal("2000"), Currency("EUR"))
    rate = Decimal("0.1")
    start = Date(2023, 1, 1)
    asof = Date(2023, 3, 31)
    result = dcc.interest(principal, rate, start, asof)
    expected = Money(Decimal("50"), Currency("EUR"))
    assert result == expected

def test_interest_returns_zero_when_dates_out_of_order():
    dcc = DCC(name="Actual/360", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 6, 30)
    asof = Date(2023, 1, 1)
    end = Date(2023, 12, 31)
    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money(Decimal("0"), Currency("USD"))
    assert result == expected

def test_interest_with_freq_passed_to_fraction_method():
    captured_freq = None
    def fraction_method(s, a, e, f):
        nonlocal captured_freq
        captured_freq = f
        return Decimal("0.3")
    dcc = DCC(name="Actual/360", altnames=set(), currencies=set(), calculate_fraction_method=fraction_method)
    principal = Money(Decimal("1500"), Currency("GBP"))
    rate = Decimal("0.04")
    start = Date(2023, 1, 1)
    asof = Date(2023, 9, 30)
    end = Date(2023, 12, 31)
    freq = Decimal("2")
    result = dcc.interest(principal, rate, start, asof, end, freq)
    expected = Money(Decimal("18"), Currency("GBP"))
    assert result == expected
    assert captured_freq == freq

def test_interest_with_zero_fraction_returns_zero_interest():
    dcc = DCC(name="Actual/360", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0"))
    principal = Money(Decimal("5000"), Currency("JPY"))
    rate = Decimal("0.02")
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 1)
    end = Date(2023, 12, 31)
    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money(Decimal("0"), Currency("JPY"))
    assert result == expected


# LLM-generated content at query #3
#--------------------------

def test_dcfc_30_e_360_basic_examples():
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result1 = round(dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14)
    result2 = round(dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14)
    result3 = round(dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14)
    result4 = round(dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14)
    assert result1 == Decimal('0.16666666666667')
    assert result2 == Decimal('0.16944444444444')
    assert result3 == Decimal('1.08333333333333')
    assert result4 == Decimal('1.33055555555556')

def test_dcfc_30_e_360_start_day_31_adjustment():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    expected = Decimal((28 - 30) + 30 * (2 - 1) + 360 * (2023 - 2023)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_asof_day_31_adjustment():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 3, 31)
    end = datetime.date(2023, 3, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    expected = Decimal((30 - 30) + 30 * (3 - 1) + 360 * (2023 - 2023)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_both_days_31_adjustment():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    end = datetime.date(2023, 3, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    expected = Decimal((30 - 30) + 30 * (3 - 1) + 360 * (2023 - 2023)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_no_adjustment():
    start = datetime.date(2023, 2, 15)
    asof = datetime.date(2023, 5, 15)
    end = datetime.date(2023, 5, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    expected = Decimal((15 - 15) + 30 * (5 - 2) + 360 * (2023 - 2023)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_cross_year():
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 1, 15)
    end = datetime.date(2023, 1, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    expected = Decimal((15 - 15) + 30 * (1 - 12) + 360 * (2023 - 2022)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_leap_year_feb_29():
    start = datetime.date(2024, 2, 28)
    asof = datetime.date(2024, 2, 29)
    end = datetime.date(2024, 2, 29)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    expected = Decimal((29 - 28) + 30 * (2 - 2) + 360 * (2024 - 2024)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_same_date():
    start = datetime.date(2023, 5, 15)
    asof = datetime.date(2023, 5, 15)
    end = datetime.date(2023, 5, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    assert result == Decimal(0)

def test_dcfc_30_e_360_negative_days():
    start = datetime.date(2023, 5, 20)
    asof = datetime.date(2023, 5, 10)
    end = datetime.date(2023, 5, 10)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    expected = Decimal((10 - 20) + 30 * (5 - 5) + 360 * (2023 - 2023)) / Decimal(360)
    assert result == expected


# LLM-generated content at query #4
#--------------------------

def test_get_date_range_with_single_day():
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 2)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 1, 1)]
    assert result == expected

def test_get_date_range_with_multiple_days():
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 5)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2), datetime.date(2023, 1, 3), datetime.date(2023, 1, 4)]
    assert result == expected

def test_get_date_range_with_zero_days():
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2023, 1, 1)
    result = list(_get_date_range(start, end))
    expected = []
    assert result == expected

def test_get_date_range_with_leap_year():
    start = datetime.date(2024, 2, 28)
    end = datetime.date(2024, 3, 2)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2024, 2, 28), datetime.date(2024, 2, 29), datetime.date(2024, 3, 1)]
    assert result == expected

def test_get_date_range_with_month_crossing():
    start = datetime.date(2023, 1, 31)
    end = datetime.date(2023, 2, 3)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 1, 31), datetime.date(2023, 2, 1), datetime.date(2023, 2, 2)]
    assert result == expected

def test_get_date_range_with_year_crossing():
    start = datetime.date(2023, 12, 30)
    end = datetime.date(2024, 1, 2)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 12, 30), datetime.date(2023, 12, 31), datetime.date(2024, 1, 1)]
    assert result == expected


# LLM-generated content at query #5
#--------------------------

def test_find_exact_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    result = registry.find("Act/Act")
    assert result == dcc

def test_find_exact_alt_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=["Actual/Actual"])
    registry.register(dcc)
    result = registry.find("Actual/Actual")
    assert result == dcc

def test_find_stripped_uppercase_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="ACT/ACT", altnames=[])
    registry.register(dcc)
    result = registry.find(" act/act ")
    assert result == dcc

def test_find_stripped_uppercase_alt_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=["ACTUAL/ACTUAL"])
    registry.register(dcc)
    result = registry.find(" actual/actual ")
    assert result == dcc

def test_find_not_registered():
    registry = DCCRegistryMachinery()
    result = registry.find("Unknown")
    assert result is None

def test_find_case_insensitive_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="ACT/ACT", altnames=[])
    registry.register(dcc)
    result = registry.find("act/act")
    assert result == dcc

def test_find_case_insensitive_alt_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=["ACTUAL/ACTUAL"])
    registry.register(dcc)
    result = registry.find("actual/actual")
    assert result == dcc

def test_find_with_whitespace_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="ACT/ACT", altnames=[])
    registry.register(dcc)
    result = registry.find("  ACT/ACT  ")
    assert result == dcc

def test_find_with_whitespace_alt_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=["ACTUAL/ACTUAL"])
    registry.register(dcc)
    result = registry.find("  ACTUAL/ACTUAL  ")
    assert result == dcc

def test_find_empty_string():
    registry = DCCRegistryMachinery()
    result = registry.find("")
    assert result is None

def test_find_whitespace_only():
    registry = DCCRegistryMachinery()
    result = registry.find("   ")
    assert result is None


# LLM-generated content at query #6
#--------------------------

def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #7
#--------------------------

def test_buffer_main_is_empty_dict():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}

def test_buffer_altn_is_empty_dict():
    registry = DCCRegistryMachinery()
    assert registry._buffer_altn == {}


# LLM-generated content at query #8
#--------------------------

def test_coupon_basic_calculation():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 1)
    freq = Decimal("2")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.5")
    assert result == expected

def test_coupon_with_eom_adjustment():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.25"))
    principal = Money(Decimal("2000"), Currency("EUR"))
    rate = Decimal("0.03")
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 3, 15)
    end = datetime.date(2020, 7, 31)
    freq = Decimal("4")
    eom = 31
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.25")
    assert result == expected

def test_coupon_zero_fraction():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0"))
    principal = Money(Decimal("500"), Currency("GBP"))
    rate = Decimal("0.02")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 7, 1)
    freq = Decimal("2")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0")
    assert result == expected

def test_coupon_fraction_greater_than_one():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("1.5"))
    principal = Money(Decimal("1500"), Currency("JPY"))
    rate = Decimal("0.01")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2021, 1, 1)
    freq = Decimal("1")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("1.5")
    assert result == expected

def test_coupon_negative_rate():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("-0.02")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 4, 1)
    end = datetime.date(2020, 7, 1)
    freq = Decimal("2")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.3")
    assert result == expected

def test_coupon_principal_zero():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.6"))
    principal = Money(Decimal("0"), Currency("EUR"))
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 1)
    freq = Decimal("2")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.6")
    assert result == expected

def test_coupon_freq_as_int():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.4"))
    principal = Money(Decimal("1200"), Currency("CAD"))
    rate = Decimal("0.04")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 5, 1)
    end = datetime.date(2020, 10, 1)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.4")
    assert result == expected

def test_coupon_eom_31_invalid_month():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.33"))
    principal = Money(Decimal("800"), Currency("AUD"))
    rate = Decimal("0.025")
    start = datetime.date(2020, 2, 29)
    asof = datetime.date(2020, 4, 30)
    end = datetime.date(2020, 8, 31)
    freq = Decimal("4")
    eom = 31
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.33")
    assert result == expected

def test_coupon_asof_equals_start():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.0"))
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 7, 1)
    freq = Decimal("2")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.0")
    assert result == expected

def test_coupon_asof_equals_end():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("1.0"))
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2020, 7, 1)
    freq = Decimal("2")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("1.0")
    assert result == expected


# LLM-generated content at query #9
#--------------------------

def test_register_successfully_adds_new_dcc():
    registry = DCCRegistryMachinery()
    dummy_dcc = DCC(name="TestDCC", altnames={"TDCC"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dummy_dcc)
    assert registry._find_strict("TestDCC") == dummy_dcc
    assert registry._find_strict("TDCC") == dummy_dcc

def test_register_raises_error_on_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dummy_dcc1 = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dummy_dcc2 = DCC(name="TestDCC", altnames={"Other"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dummy_dcc1)
    try:
        registry.register(dummy_dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'TestDCC' is already registered"

def test_register_raises_error_on_duplicate_altname_in_main_buffer():
    registry = DCCRegistryMachinery()
    dummy_dcc1 = DCC(name="FirstDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dummy_dcc2 = DCC(name="SecondDCC", altnames={"FirstDCC"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dummy_dcc1)
    try:
        registry.register(dummy_dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'SecondDCC' is already registered"

def test_register_raises_error_on_duplicate_altname_in_alt_buffer():
    registry = DCCRegistryMachinery()
    dummy_dcc1 = DCC(name="FirstDCC", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dummy_dcc2 = DCC(name="SecondDCC", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dummy_dcc1)
    try:
        registry.register(dummy_dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'SecondDCC' is already registered"

def test_register_adds_all_altnames_to_alt_buffer():
    registry = DCCRegistryMachinery()
    altnames = {"Alt1", "Alt2", "Alt3"}
    dummy_dcc = DCC(name="TestDCC", altnames=altnames, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dummy_dcc)
    for alt in altnames:
        assert registry._find_strict(alt) == dummy_dcc

def test_register_does_not_affect_existing_registry_when_error_occurs():
    registry = DCCRegistryMachinery()
    dummy_dcc1 = DCC(name="FirstDCC", altnames={"Alt1"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dummy_dcc2 = DCC(name="SecondDCC", altnames={"FirstDCC"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dummy_dcc1)
    try:
        registry.register(dummy_dcc2)
    except TypeError:
        pass
    assert registry._find_strict("FirstDCC") == dummy_dcc1
    assert registry._find_strict("Alt1") == dummy_dcc1
    assert registry._find_strict("SecondDCC") is None


# LLM-generated content at query #10
#--------------------------

def test_dcfc_30_e_plus_360_basic_examples():
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    ex1_end = ex1_asof
    result1 = dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_end)
    assert round(result1, 14) == Decimal('0.16666666666667')
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    ex2_end = ex2_asof
    result2 = dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_end)
    assert round(result2, 14) == Decimal('0.16944444444444')
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    ex3_end = ex3_asof
    result3 = dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_end)
    assert round(result3, 14) == Decimal('1.08333333333333')
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    ex4_end = ex4_asof
    result4 = dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_end)
    assert round(result4, 14) == Decimal('1.33333333333333')

def test_dcfc_30_e_plus_360_start_day_31_adjustment():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=end)
    expected = Decimal((28 - 30) + 30 * (2 - 1) + 360 * (2023 - 2023)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_plus_360_asof_day_31_adjustment():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    end = asof
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=end)
    adjusted_asof = datetime.date(2023, 3, 1)
    expected = Decimal((adjusted_asof.day - start.day) + 30 * (adjusted_asof.month - start.month) + 360 * (adjusted_asof.year - start.year)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_plus_360_both_days_31_adjustment():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    end = asof
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=end)
    adjusted_start = datetime.date(2023, 1, 30)
    adjusted_asof = datetime.date(2023, 4, 1)
    expected = Decimal((adjusted_asof.day - adjusted_start.day) + 30 * (adjusted_asof.month - adjusted_start.month) + 360 * (adjusted_asof.year - adjusted_start.year)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_plus_360_no_adjustment():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=end)
    expected = Decimal((28 - 30) + 30 * (2 - 1) + 360 * (2023 - 2023)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_plus_360_cross_year():
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 1, 15)
    end = asof
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=end)
    expected = Decimal((15 - 15) + 30 * (1 - 12) + 360 * (2023 - 2022)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_plus_360_same_date():
    start = datetime.date(2023, 5, 15)
    asof = start
    end = start
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=end)
    assert result == Decimal(0)

def test_dcfc_30_e_plus_360_leap_year_feb29():
    start = datetime.date(2024, 2, 28)
    asof = datetime.date(2024, 2, 29)
    end = asof
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=end)
    expected = Decimal((29 - 28) + 30 * (2 - 2) + 360 * (2024 - 2024)) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_plus_360_leap_year_feb29_asof_31_adjustment():
    start = datetime.date(2024, 2, 29)
    asof = datetime.date(2024, 3, 31)
    end = asof
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=end)
    adjusted_asof = datetime.date(2024, 4, 1)
    expected = Decimal((adjusted_asof.day - start.day) + 30 * (adjusted_asof.month - start.month) + 360 * (adjusted_asof.year - start.year)) / Decimal(360)
    assert result == expected


# LLM-generated content at query #11
#--------------------------

def test_last_payment_date_annual_frequency():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_same_year_annual():
    start = datetime.date(2015, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_semiannual_frequency():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 7, 1)
    assert result == expected

def test_last_payment_date_semiannual_before_midyear():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 8, 31)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 7, 1)
    assert result == expected

def test_last_payment_date_semiannual_early_year():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 4, 30)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_annual_start_midyear():
    start = datetime.date(2014, 6, 1)
    asof = datetime.date(2015, 4, 30)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2014, 6, 1)
    assert result == expected

def test_last_payment_date_quarterly_frequency():
    start = datetime.date(2008, 7, 7)
    asof = datetime.date(2015, 10, 6)
    frequency = 4
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 7, 7)
    assert result == expected

def test_last_payment_date_annual_december_start():
    start = datetime.date(2014, 12, 9)
    asof = datetime.date(2015, 12, 4)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2014, 12, 9)
    assert result == expected

def test_last_payment_date_semiannual_december_start():
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2016, 1, 6)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 12, 15)
    assert result == expected

def test_last_payment_date_semiannual_december_end():
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2015, 12, 31)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 12, 15)
    assert result == expected

def test_last_payment_date_with_eom_override():
    start = datetime.date(2014, 1, 31)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = 15
    result = _last_payment_date(start, asof, frequency, eom)
    expected = datetime.date(2015, 1, 15)
    assert result == expected

def test_last_payment_date_monthly_frequency():
    start = datetime.date(2014, 3, 10)
    asof = datetime.date(2015, 12, 20)
    frequency = 12
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 12, 10)
    assert result == expected

def test_last_payment_date_biannual_frequency():
    start = datetime.date(2014, 2, 28)
    asof = datetime.date(2015, 12, 31)
    frequency = 6
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 8, 28)
    assert result == expected

def test_last_payment_date_edge_case_negative_year():
    start = datetime.date(1, 1, 1)
    asof = datetime.date(2, 1, 1)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(1, 1, 1)
    assert result == expected

def test_last_payment_date_invalid_date_handling():
    start = datetime.date(2014, 2, 30)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2014, 2, 28)
    assert result == expected

def test_last_payment_date_frequency_zero_division():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 0
    try:
        _last_payment_date(start, asof, frequency)
        assert False
    except ZeroDivisionError:
        assert True

def test_last_payment_date_frequency_decimal():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = Decimal('1')
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 1, 1)
    assert result == expected


# LLM-generated content at query #12
#--------------------------

```python
def test_dcfc_nl_365_handles_leap_day_correctly():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_nl_365(start=start, asof=asof, end=end)
    expected = Decimal('0.16986301369863')
    assert round(result, 14) == expected


# LLM-generated content at query #13
#--------------------------

def test_dcfc_30_360_german_basic():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

def test_dcfc_30_360_german_leap_year():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

def test_dcfc_30_360_german_31_day_start():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_360_german_feb_start():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    expected = Decimal('1.33055555555556')
    assert round(result, 14) == expected

def test_dcfc_30_360_german_same_date():
    start = datetime.date(2023, 5, 15)
    asof = start
    end = start
    result = dcfc_30_360_german(start, asof, end)
    assert result == Decimal('0')

def test_dcfc_30_360_german_start_31_adjusted():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    d1 = 30
    d2 = asof.day
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = nod / Decimal(360)
    assert result == expected

def test_dcfc_30_360_german_asof_31_adjusted():
    start = datetime.date(2023, 3, 15)
    asof = datetime.date(2023, 4, 30)
    end = datetime.date(2023, 5, 15)
    result = dcfc_30_360_german(start, asof, end)
    d1 = start.day
    d2 = 30
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = nod / Decimal(360)
    assert result == expected

def test_dcfc_30_360_german_feb_last_day_start():
    start = datetime.date(2023, 2, 28)
    asof = datetime.date(2023, 3, 31)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    d1 = 30
    d2 = 30
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = nod / Decimal(360)
    assert result == expected

def test_dcfc_30_360_german_feb_last_day_asof_not_end():
    start = datetime.date(2023, 1, 15)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 3, 15)
    result = dcfc_30_360_german(start, asof, end)
    d1 = start.day
    d2 = 30
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = nod / Decimal(360)
    assert result == expected

def test_dcfc_30_360_german_feb_last_day_asof_is_end():
    start = datetime.date(2023, 1, 15)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_german(start, asof, end)
    d1 = start.day
    d2 = asof.day
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = nod / Decimal(360)
    assert result == expected


# LLM-generated content at query #14
#--------------------------

def test_register_new_dcc_successfully():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TDCC"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc

def test_register_duplicate_main_name_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="TestDCC", altnames={"Alt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'TestDCC' is already registered"

def test_register_duplicate_altname_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="FirstDCC", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="SecondDCC", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'SecondDCC' is already registered"

def test_register_altname_conflict_with_main_name_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="MainDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="AnotherDCC", altnames={"MainDCC"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'AnotherDCC' is already registered"

def test_register_main_name_conflict_with_altname_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="FirstDCC", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="AltName", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'AltName' is already registered"

def test_register_multiple_altnames_successfully():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"Alt1", "Alt2"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("Alt1") == dcc
    assert registry._find_strict("Alt2") == dcc

def test_register_empty_altnames_successfully():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("NonExistent") is None


# LLM-generated content at query #15
#--------------------------

def test_dcfc_act_365_a_basic():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    expected = Decimal('0.16986301369863')
    assert round(result, 14) == expected

def test_dcfc_act_365_a_leap_year():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    expected = Decimal('0.17213114754098')
    assert round(result, 14) == expected

def test_dcfc_act_365_a_long_period():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    expected = Decimal('1.08196721311475')
    assert round(result, 14) == expected

def test_dcfc_act_365_a_another_long_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    expected = Decimal('1.32513661202186')
    assert round(result, 14) == expected

def test_dcfc_act_365_a_same_day():
    start = datetime.date(2017, 1, 1)
    asof = start
    end = start
    result = dcfc_act_365_a(start, asof, end)
    assert result == Decimal('0')

def test_dcfc_act_365_a_one_day():
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 2)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    expected = Decimal('1') / Decimal(365)
    assert result == expected

def test_dcfc_act_365_a_non_leap_year():
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2019, 12, 31)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    expected = Decimal('364') / Decimal(365)
    assert result == expected

def test_dcfc_act_365_a_leap_year_full():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    expected = Decimal('365') / Decimal(366)
    assert result == expected

def test_dcfc_act_365_a_leap_day_in_range():
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    expected = Decimal('2') / Decimal(366)
    assert result == expected

def test_dcfc_act_365_a_no_leap_day_in_range():
    start = datetime.date(2021, 2, 28)
    asof = datetime.date(2021, 3, 1)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    expected = Decimal('1') / Decimal(365)
    assert result == expected


# LLM-generated content at query #16
#--------------------------

def test_register_raises_error_when_main_name_already_registered():
    dcc1 = DCC(name="Act/Act", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"))
    dcc2 = DCC(name="Act/Act", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.2"))
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'Act/Act' is already registered"


# LLM-generated content at query #17
#--------------------------

```python
def test_dcfc_30_e_360_start_day_31_adjusts_to_30():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected


# LLM-generated content at query #18
#--------------------------

def test_dcfc_30_e_360_example1():
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_example2():
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_example3():
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_example4():
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    expected = Decimal('1.33055555555556')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_start_day_31():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    adjusted_start = datetime.date(2023, 1, 30)
    nod = (asof.day - adjusted_start.day) + 30 * (asof.month - adjusted_start.month) + 360 * (asof.year - adjusted_start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_asof_day_31():
    start = datetime.date(2023, 1, 28)
    asof = datetime.date(2023, 2, 31)
    end = datetime.date(2023, 2, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    adjusted_asof = datetime.date(2023, 2, 30)
    nod = (adjusted_asof.day - start.day) + 30 * (adjusted_asof.month - start.month) + 360 * (adjusted_asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_both_days_31():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 31)
    end = datetime.date(2023, 2, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    adjusted_start = datetime.date(2023, 1, 30)
    adjusted_asof = datetime.date(2023, 2, 30)
    nod = (adjusted_asof.day - adjusted_start.day) + 30 * (adjusted_asof.month - adjusted_start.month) + 360 * (adjusted_asof.year - adjusted_start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_no_adjustment():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    nod = (asof.day - start.day) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_negative_days():
    start = datetime.date(2023, 2, 28)
    asof = datetime.date(2023, 1, 28)
    end = datetime.date(2023, 1, 28)
    result = dcfc_30_e_360(start=start, asof=asof, end=asof)
    adjusted_start = datetime.date(2023, 2, 28)
    adjusted_asof = datetime.date(2023, 1, 28)
    nod = (adjusted_asof.day - adjusted_start.day) + 30 * (adjusted_asof.month - adjusted_start.month) + 360 * (adjusted_asof.year - adjusted_start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_same_date():
    start = datetime.date(2023, 5, 15)
    asof = datetime.date(2023, 5, 15)
    end = datetime.date(2023, 5, 15)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    assert result == Decimal(0)

def test_dcfc_30_e_360_leap_year():
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    nod = (asof.day - start.day) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_year_boundary():
    start = datetime.date(2022, 12, 31)
    asof = datetime.date(2023, 1, 31)
    end = datetime.date(2023, 1, 31)
    result = dcfc_30_e_360(start=start, asof=asof, end=end)
    adjusted_start = datetime.date(2022, 12, 30)
    adjusted_asof = datetime.date(2023, 1, 30)
    nod = (adjusted_asof.day - adjusted_start.day) + 30 * (adjusted_asof.month - adjusted_start.month) + 360 * (adjusted_asof.year - adjusted_start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected


# LLM-generated content at query #19
#--------------------------

def test_is_last_day_of_month_true():
    date = Date(2023, 1, 31)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_false():
    date = Date(2023, 1, 15)
    result = _is_last_day_of_month(date)
    assert result == False

def test_is_last_day_of_month_february_non_leap():
    date = Date(2023, 2, 28)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_february_leap():
    date = Date(2024, 2, 29)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_december():
    date = Date(2023, 12, 31)
    result = _is_last_day_of_month(date)
    assert result == True

def test_is_last_day_of_month_april():
    date = Date(2023, 4, 30)
    result = _is_last_day_of_month(date)
    assert result == True


# LLM-generated content at query #20
#--------------------------

def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #21
#--------------------------

def test_find_strips_and_uppercases_name_as_last_resort():
    registry = DCCRegistryMachinery()
    mock_dcc = DCC(name="ACT/ACT", altnames=[])
    registry._buffer_main["ACT/ACT"] = mock_dcc
    result = registry.find("  act/act  ")
    assert result is mock_dcc


# LLM-generated content at query #22
#--------------------------

def test_buffer_main_is_empty_dict_on_init():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}


# LLM-generated content at query #23
#--------------------------

def test_register_raises_type_error_when_altname_already_registered():
    dcc1 = DCC(name="Test1", altnames={"Alt1"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0"))
    dcc2 = DCC(name="Test2", altnames={"Alt1"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0"))
    registry = DCCRegistryMachinery()
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'Test2' is already registered"


# LLM-generated content at query #24
#--------------------------

def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #25
#--------------------------

def test_register_raises_type_error_when_altname_conflict():
    from collections import namedtuple
    from decimal import Decimal
    from typing import Set, Optional
    Currency = namedtuple('Currency', ['code'])
    DCFC = type(lambda start, asof, end, freq: Decimal('0'))
    DCC = namedtuple('DCC', ['name', 'altnames', 'currencies', 'calculate_fraction_method'])
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name='Act/Act', altnames={'Actual/Actual'}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal('0'))
    registry.register(dcc1)
    dcc2 = DCC(name='Act/360', altnames={'Actual/360'}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal('0'))
    registry.register(dcc2)
    dcc3 = DCC(name='NewDCC', altnames={'Actual/360'}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal('0'))
    try:
        registry.register(dcc3)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'NewDCC' is already registered"


