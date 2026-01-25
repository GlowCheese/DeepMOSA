####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_register_successfully_adds_dcc_to_main_buffer():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"AltTestDCC"}, currencies={}, calculate_fraction_method=lambda *args: Decimal(0.5))
    registry.register(dcc)
    assert registry._buffer_main["TestDCC"] == dcc

def test_register_successfully_adds_dcc_to_alternative_buffer():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"AltTestDCC"}, currencies={}, calculate_fraction_method=lambda *args: Decimal(0.5))
    registry.register(dcc)
    assert registry._buffer_altn["AltTestDCC"] == dcc

def test_register_raises_typeerror_for_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC", altnames={}, currencies={}, calculate_fraction_method=lambda *args: Decimal(0.5))
    dcc2 = DCC(name="TestDCC", altnames={}, currencies={}, calculate_fraction_method=lambda *args: Decimal(0.5))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)

def test_register_raises_typeerror_for_duplicate_alternative_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC", altnames={"AltTestDCC"}, currencies={}, calculate_fraction_method=lambda *args: Decimal(0.5))
    dcc2 = DCC(name="AnotherDCC", altnames={"AltTestDCC"}, currencies={}, calculate_fraction_method=lambda *args: Decimal(0.5))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #2
#--------------------------

```python
def test_DCCRegistryMachinery_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert hasattr(registry, '_buffer_main')
    assert isinstance(registry._buffer_main, dict)
    assert len(registry._buffer_main) == 0
    assert hasattr(registry, '_buffer_altn')
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_init_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #4
#--------------------------

```python
def test_last_payment_date_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1) == datetime.date(2014, 12, 9)

def test_last_payment_date_semi_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 7, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2) == datetime.date(2015, 7, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2) == datetime.date(2015, 12, 15)
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 12, 15)

def test_last_payment_date_quarterly_frequency():
    assert _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4) == datetime.date(2015, 7, 7)

def test_last_payment_date_same_start_and_asof_month():
    assert _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2014, 6, 1)


# LLM-generated content at query #5
#--------------------------

```python
def test_get_date_range_empty():
    start = Date(2020, 1, 1)
    end = Date(2020, 1, 1)
    assert list(_get_date_range(start, end)) == []

def test_get_date_range_single_day():
    start = Date(2020, 1, 1)
    end = Date(2020, 1, 2)
    assert list(_get_date_range(start, end)) == [Date(2020, 1, 1)]

def test_get_date_range_multiple_days():
    start = Date(2020, 1, 1)
    end = Date(2020, 1, 4)
    assert list(_get_date_range(start, end)) == [Date(2020, 1, 1), Date(2020, 1, 2), Date(2020, 1, 3)]

def test_get_date_range_year_boundary():
    start = Date(2019, 12, 30)
    end = Date(2020, 1, 3)
    assert list(_get_date_range(start, end)) == [Date(2019, 12, 30), Date(2019, 12, 31), Date(2020, 1, 1), Date(2020, 1, 2)]


# LLM-generated content at query #6
#--------------------------

```python
def test_register_raises_typeerror_for_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc1)
    with pytest.raises(TypeError) as exc_info:
        registry.register(dcc2)
    assert str(exc_info.value) == "Day count convention 'Test' is already registered"


# LLM-generated content at query #7
#--------------------------

```python
def test_calculate_daily_fraction_with_valid_dates():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 365))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(1 / 365)

def test_calculate_daily_fraction_with_asof_equal_to_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 365))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 1)
    end = Date(2023, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(1 / 365)

def test_calculate_daily_fraction_with_asof_before_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 365))
    start = Date(2023, 1, 1)
    asof = Date(2022, 12, 31)
    end = Date(2023, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(0)

def test_calculate_daily_fraction_with_asof_after_end():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 365))
    start = Date(2023, 1, 1)
    asof = Date(2024, 1, 1)
    end = Date(2023, 12, 31)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(0)

def test_calculate_daily_fraction_with_custom_frequency():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 365) * f if f else Decimal((a - s).days / 365))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 12, 31)
    freq = Decimal(2)
    result = dcc.calculate_daily_fraction(start, asof, end, freq)
    assert result == Decimal(2 / 365)


# LLM-generated content at query #8
#--------------------------

```python
def test_calculate_daily_fraction_asof_minus_1_not_less_than_start():
    dcc = DCC("test", set(), set(), lambda s, a, e, f: Decimal(1))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(0)


# LLM-generated content at query #9
#--------------------------

```python
def test_DCCRegistryMachinery_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #10
#--------------------------

```python
def test_construct_date_valid_date():
    result = _construct_date(2023, 5, 15)
    assert result.year == 2023
    assert result.month == 5
    assert result.day == 15

def test_construct_date_invalid_day():
    result = _construct_date(2023, 2, 30)
    assert result.year == 2023
    assert result.month == 2
    assert result.day == 28

def test_construct_date_invalid_month():
    try:
        _construct_date(2023, 13, 1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_construct_date_invalid_year():
    try:
        _construct_date(0, 1, 1)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_construct_date_negative_values():
    try:
        _construct_date(-1, -1, -1)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_dcfc_30_360_german_basic_cases():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #12
#--------------------------

```python
def test_dcfc_30_360_german_example_1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_german(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_german_example_2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_german(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_german_example_3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_german(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_german_example_4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_german(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #13
#--------------------------

```python
def test_interest_calculates_accrued_interest_correctly():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(360)
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)
    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money(1000 * 0.05 * 151 / 360, Currency("USD"))
    assert result == expected

def test_interest_uses_asof_as_end_when_end_is_none():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(360)
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    result = dcc.interest(principal, rate, start, asof)
    expected = Money(1000 * 0.05 * 151 / 360, Currency("USD"))
    assert result == expected

def test_interest_returns_zero_when_asof_before_start():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(360)
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2022, 12, 1)
    end = Date(2023, 12, 31)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == ZERO

def test_interest_returns_zero_when_asof_after_end():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(360)
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2024, 1, 1)
    end = Date(2023, 12, 31)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == ZERO


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_evaluates_to_false():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = None
    assert not (eom or start.day)


# LLM-generated content at query #15
#--------------------------

```python
def test_dcfc_30_360_isda_example_1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_isda_example_2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_isda_example_3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_isda_example_4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #16
#--------------------------

```python
def test_find_existing_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: Decimal('0.5'))
    registry.register(dcc)
    assert registry.find("Act/Act") == dcc
    assert registry.find("ACT/ACT") == dcc
    assert registry.find("act/act") == dcc

def test_find_nonexistent_dcc():
    registry = DCCRegistryMachinery()
    assert registry.find("NonExistent") is None
    assert registry.find("  ") is None


# LLM-generated content at query #17
#--------------------------

```python
def test_dcfc_30_360_us_example_1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_us_example_2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_us_example_3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_us_example_4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #18
#--------------------------

```python
def test_dcfc_nl_365_with_no_leap_day():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16986301369863')

def test_dcfc_nl_365_with_leap_day():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16986301369863')

def test_dcfc_nl_365_long_period():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08219178082192')

def test_dcfc_nl_365_cross_year():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_nl_365(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.32602739726027')


# LLM-generated content at query #19
#--------------------------

```python
def test_register_raises_typeerror_for_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda *args: ZERO)
    registry.register(dcc1)
    dcc2 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda *args: ZERO)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #20
#--------------------------

```python
def test_dcfc_30_360_us_example_1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_us_example_2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_us_example_3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_us_example_4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #21
#--------------------------

```python
def test_find_returns_correct_dcc_with_stripped_uppercase_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT", "Actual/Actual"])
    registry.register(dcc)
    assert registry.find(" act/act ") == dcc


# LLM-generated content at query #22
#--------------------------

```python
def test_init_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #23
#--------------------------

```python
def test_dcfc_30_360_german_asof_day_31_or_last_day_of_feb():
    asof_date = datetime.date(2023, 2, 28)
    end_date = datetime.date(2023, 3, 1)
    start_date = datetime.date(2023, 1, 1)
    result = dcfc_30_360_german(start_date, asof_date, end_date)
    assert result == Decimal('57') / Decimal('360')


# LLM-generated content at query #24
#--------------------------

```python
def test_calculate_daily_fraction_when_asof_minus_1_less_than_start():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    start = Date(2023, 1, 2)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal("0.5")


# LLM-generated content at query #25
#--------------------------

```python
def test_dcc_registry_machinery_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #26
#--------------------------

```python
def test_calculate_fraction_with_valid_dates():
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)
    assert dcc.calculate_fraction(start, asof, end) == Decimal("0.5")

def test_calculate_fraction_with_invalid_dates():
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    start = Date(2023, 1, 1)
    asof = Date(2022, 6, 1)
    end = Date(2023, 12, 31)
    assert dcc.calculate_fraction(start, asof, end) == ZERO

def test_calculate_fraction_with_frequency():
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: f or Decimal("0.5")
    )
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)
    freq = Decimal("0.75")
    assert dcc.calculate_fraction(start, asof, end, freq) == freq


# LLM-generated content at query #27
#--------------------------

```python
def test_dcfc_act_act_example_1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16942884946478')

def test_dcfc_act_act_example_2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.17216108990194')

def test_dcfc_act_act_example_3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08243131970956')

def test_dcfc_act_act_example_4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32625945055768')


# LLM-generated content at query #28
#--------------------------

```python
def test_d1_equals_31():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    assert dcfc_30_360_us(start=start, asof=asof, end=asof) == Decimal('1.33333333333333')


# LLM-generated content at query #29
#--------------------------

```python
def test_dcfc_act_365_a_without_leap_day():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 6, 30)
    end = datetime.date(2021, 12, 31)
    expected = Decimal("0.5")
    assert dcfc_act_365_a(start, asof, end) == expected

def test_dcfc_act_365_a_with_leap_day():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2020, 12, 31)
    expected = Decimal("0.5013698630137")
    assert round(dcfc_act_365_a(start, asof, end), 14) == expected

def test_dcfc_act_365_a_full_year_without_leap():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 12, 31)
    end = datetime.date(2021, 12, 31)
    expected = Decimal("1.0")
    assert dcfc_act_365_a(start, asof, end) == expected

def test_dcfc_act_365_a_full_year_with_leap():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    expected = Decimal("1.0027397260274")
    assert round(dcfc_act_365_a(start, asof, end), 14) == expected

def test_dcfc_act_365_a_partial_period():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 1, 15)
    end = datetime.date(2021, 1, 31)
    expected = Decimal("0.0410958904109589")
    assert round(dcfc_act_365_a(start, asof, end), 14) == expected

def test_dcfc_act_365_a_same_start_and_asof():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 12, 31)
    expected = Decimal("0.0")
    assert dcfc_act_365_a(start, asof, end) == expected

def test_dcfc_act_365_a_invalid_date_range():
    start = datetime.date(2021, 12, 31)
    asof = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 6, 30)
    expected = Decimal("0.0")
    assert dcfc_act_365_a(start, asof, end) == expected


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_at_line_34_evaluates_to_false():
    start_date = datetime.date(2007, 12, 31)
    asof_date = datetime.date(2008, 1, 31)
    assert not _is_last_day_of_month(asof_date)


# LLM-generated content at query #31
#--------------------------

```python
def test_construct_date_with_valid_date():
    result = _construct_date(2023, 5, 15)
    assert isinstance(result, datetime.date)
    assert result.year == 2023
    assert result.month == 5
    assert result.day == 15


# LLM-generated content at query #32
#--------------------------

```python
def test_register_altname_conflict():
    registry = DCCRegistryMachinery()
    dcc1 = DCC("Test1", {"Test1"}, set(), lambda s, a, e, f: Decimal(1))
    dcc2 = DCC("Test2", {"Test1"}, set(), lambda s, a, e, f: Decimal(1))
    registry.register(dcc1)
    assert registry._is_registered("Test1")
    try:
        registry.register(dcc2)
    except TypeError as e:
        assert str(e) == "Day count convention 'Test2' is already registered"


# LLM-generated content at query #33
#--------------------------

```python
def test_next_payment_date_annual_no_eom():
    assert _next_payment_date(datetime.date(2014, 1, 1), 1, None) == datetime.date(2015, 1, 1)

def test_next_payment_date_annual_with_eom():
    assert _next_payment_date(datetime.date(2014, 1, 1), 1, 15) == datetime.date(2015, 1, 15)

def test_next_payment_date_semiannual_no_eom():
    assert _next_payment_date(datetime.date(2014, 1, 1), 2, None) == datetime.date(2014, 7, 1)

def test_next_payment_date_semiannual_with_eom():
    assert _next_payment_date(datetime.date(2014, 1, 1), 2, 15) == datetime.date(2014, 7, 15)

def test_next_payment_date_quarterly_no_eom():
    assert _next_payment_date(datetime.date(2014, 1, 1), 4, None) == datetime.date(2014, 4, 1)

def test_next_payment_date_quarterly_with_eom():
    assert _next_payment_date(datetime.date(2014, 1, 1), 4, 15) == datetime.date(2014, 4, 15)

def test_next_payment_date_monthly_no_eom():
    assert _next_payment_date(datetime.date(2014, 1, 1), 12, None) == datetime.date(2014, 2, 1)

def test_next_payment_date_monthly_with_eom():
    assert _next_payment_date(datetime.date(2014, 1, 1), 12, 15) == datetime.date(2014, 2, 15)

def test_next_payment_date_invalid_eom():
    assert _next_payment_date(datetime.date(2014, 1, 31), 1, 31) == datetime.date(2015, 1, 31)

def test_next_payment_date_february_eom():
    assert _next_payment_date(datetime.date(2014, 1, 31), 1, 29) == datetime.date(2015, 1, 31)


# LLM-generated content at query #34
#--------------------------

```python
def test_last_payment_date_predicate():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)


# LLM-generated content at query #35
#--------------------------

```python
def test_dcfc_30_360_german_asof_day_31_or_last_day_of_feb():
    asof_day_31 = datetime.date(2023, 1, 31)
    asof_last_day_of_feb = datetime.date(2023, 2, 28)
    end_date = datetime.date(2023, 3, 1)
    start_date = datetime.date(2023, 1, 1)
    assert dcfc_30_360_german(start_date, asof_day_31, end_date) == Decimal('30') / Decimal('360')
    assert dcfc_30_360_german(start_date, asof_last_day_of_feb, end_date) == Decimal('28') / Decimal('360')


# LLM-generated content at query #36
#--------------------------

```python
def test_construct_date_predicate_false():
    assert not (year <= 0 or month <= 0 or day <= 0)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_register_successful():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Test", altnames={"TestAlt"}, currencies={Currency("USD")}, calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("Test") == dcc
    assert registry._find_strict("TestAlt") == dcc

def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={"TestAlt"}, currencies={Currency("USD")}, calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="Test", altnames={"TestAlt2"}, currencies={Currency("EUR")}, calculate_fraction_method=lambda s, a, e, f: Decimal("0.6"))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)

def test_register_duplicate_alt_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={"TestAlt"}, currencies={Currency("USD")}, calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="Test2", altnames={"TestAlt"}, currencies={Currency("EUR")}, calculate_fraction_method=lambda s, a, e, f: Decimal("0.6"))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #2
#--------------------------

```python
def test_DCCRegistryMachinery_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #3
#--------------------------

```python
def test_register_successful():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Test", altnames={"TestAlt"}, currencies={Currency("USD")}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc)
    assert registry._find_strict("Test") == dcc
    assert registry._find_strict("TestAlt") == dcc

def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={"TestAlt"}, currencies={Currency("USD")}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC(name="Test", altnames={"TestAlt2"}, currencies={Currency("EUR")}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.6))
    registry.register(dcc1)
    with pytest.raises(TypeError, match="Day count convention 'Test' is already registered"):
        registry.register(dcc2)

def test_register_duplicate_alternative_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={"TestAlt"}, currencies={Currency("USD")}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC(name="Test2", altnames={"TestAlt"}, currencies={Currency("EUR")}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.6))
    registry.register(dcc1)
    with pytest.raises(TypeError, match="Day count convention 'Test2' is already registered"):
        registry.register(dcc2)


# LLM-generated content at query #4
#--------------------------

```python
def test_dcc_registry_machinery_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #5
#--------------------------

```python
def test_dcfc_30_e_plus_360_example_1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

def test_dcfc_30_e_plus_360_example_2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

def test_dcfc_30_e_plus_360_example_3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

def test_dcfc_30_e_plus_360_example_4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #6
#--------------------------

```python
def test_empty_range():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 1)
    assert list(_get_date_range(start, end)) == []

def test_single_day_range():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 2)
    assert list(_get_date_range(start, end)) == [Date(2023, 1, 1)]

def test_multi_day_range():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 5)
    expected = [Date(2023, 1, 1), Date(2023, 1, 2), Date(2023, 1, 3), Date(2023, 1, 4)]
    assert list(_get_date_range(start, end)) == expected

def test_year_boundary():
    start = Date(2022, 12, 30)
    end = Date(2023, 1, 3)
    expected = [Date(2022, 12, 30), Date(2022, 12, 31), Date(2023, 1, 1), Date(2023, 1, 2)]
    assert list(_get_date_range(start, end)) == expected

def test_leap_year():
    start = Date(2020, 2, 27)
    end = Date(2020, 3, 2)
    expected = [Date(2020, 2, 27), Date(2020, 2, 28), Date(2020, 2, 29), Date(2020, 3, 1)]
    assert list(_get_date_range(start, end)) == expected


# LLM-generated content at query #7
#--------------------------

```python
def test_dcfc_act_act_example_1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16942884946478')

def test_dcfc_act_act_example_2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.17216108990194')

def test_dcfc_act_act_example_3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08243131970956')

def test_dcfc_act_act_example_4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32625945055768')


# LLM-generated content at query #8
#--------------------------

```python
def test_find_existing_dcc_by_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: 1.0)
    registry.register(dcc)
    assert registry.find("Act/Act") == dcc

def test_find_existing_dcc_by_alternative_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: 1.0)
    registry.register(dcc)
    assert registry.find("ACT/ACT") == dcc

def test_find_existing_dcc_case_insensitive():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: 1.0)
    registry.register(dcc)
    assert registry.find("act/act") == dcc

def test_find_existing_dcc_with_whitespace():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: 1.0)
    registry.register(dcc)
    assert registry.find("  Act/Act  ") == dcc

def test_find_non_existing_dcc():
    registry = DCCRegistryMachinery()
    assert registry.find("NonExistent") is None


# LLM-generated content at query #9
#--------------------------

```python
def test_dcfc_act_act_example1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16942884946478')

def test_dcfc_act_act_example2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.17216108990194')

def test_dcfc_act_act_example3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08243131970956')

def test_dcfc_act_act_example4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32625945055768')


# LLM-generated content at query #10
#--------------------------

```python
def test_last_day_of_february_in_leap_year():
    date = Date(2020, 2, 29)
    assert _is_last_day_of_month(date) is True

def test_last_day_of_february_in_non_leap_year():
    date = Date(2021, 2, 28)
    assert _is_last_day_of_month(date) is True

def test_last_day_of_april():
    date = Date(2021, 4, 30)
    assert _is_last_day_of_month(date) is True

def test_last_day_of_december():
    date = Date(2021, 12, 31)
    assert _is_last_day_of_month(date) is True

def test_not_last_day_of_month():
    date = Date(2021, 5, 15)
    assert _is_last_day_of_month(date) is False


# LLM-generated content at query #11
#--------------------------

```python
def test_dcfc_30_e_360_example_1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

def test_dcfc_30_e_360_example_2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

def test_dcfc_30_e_360_example_3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

def test_dcfc_30_e_360_example_4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33055555555556')


# LLM-generated content at query #12
#--------------------------

```python
def test_get_date_range_basic():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 5)
    result = list(_get_date_range(start, end))
    expected = [Date(2023, 1, 1), Date(2023, 1, 2), Date(2023, 1, 3), Date(2023, 1, 4)]
    assert result == expected

def test_get_date_range_single_day():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 2)
    result = list(_get_date_range(start, end))
    expected = [Date(2023, 1, 1)]
    assert result == expected

def test_get_date_range_empty():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 1)
    result = list(_get_date_range(start, end))
    expected = []
    assert result == expected

def test_get_date_range_year_boundary():
    start = Date(2022, 12, 30)
    end = Date(2023, 1, 3)
    result = list(_get_date_range(start, end))
    expected = [Date(2022, 12, 30), Date(2022, 12, 31), Date(2023, 1, 1), Date(2023, 1, 2)]
    assert result == expected

def test_get_date_range_month_boundary():
    start = Date(2023, 1, 30)
    end = Date(2023, 2, 3)
    result = list(_get_date_range(start, end))
    expected = [Date(2023, 1, 30), Date(2023, 1, 31), Date(2023, 2, 1), Date(2023, 2, 2)]
    assert result == expected


# LLM-generated content at query #13
#--------------------------

```python
def test_dcfc_30_360_us_basic_cases():
    assert round(dcfc_30_360_us(
        start=datetime.date(2007, 12, 28),
        asof=datetime.date(2008, 2, 28),
        end=datetime.date(2008, 2, 28)
    ), 14) == Decimal('0.16666666666667')

    assert round(dcfc_30_360_us(
        start=datetime.date(2007, 12, 28),
        asof=datetime.date(2008, 2, 29),
        end=datetime.date(2008, 2, 29)
    ), 14) == Decimal('0.16944444444444')

    assert round(dcfc_30_360_us(
        start=datetime.date(2007, 10, 31),
        asof=datetime.date(2008, 11, 30),
        end=datetime.date(2008, 11, 30)
    ), 14) == Decimal('1.08333333333333')

    assert round(dcfc_30_360_us(
        start=datetime.date(2008, 2, 1),
        asof=datetime.date(2009, 5, 31),
        end=datetime.date(2009, 5, 31)
    ), 14) == Decimal('1.33333333333333')

def test_dcfc_30_360_us_edge_cases():
    assert dcfc_30_360_us(
        start=datetime.date(2020, 1, 31),
        asof=datetime.date(2020, 2, 28),
        end=datetime.date(2020, 2, 28)
    ) == Decimal('0.0')

    assert dcfc_30_360_us(
        start=datetime.date(2020, 1, 1),
        asof=datetime.date(2020, 1, 31),
        end=datetime.date(2020, 1, 31)
    ) == Decimal('30') / Decimal('360')

    assert dcfc_30_360_us(
        start=datetime.date(2020, 2, 29),
        asof=datetime.date(2020, 3, 31),
        end=datetime.date(2020, 3, 31)
    ) == Decimal('30') / Decimal('360')

def test_dcfc_30_360_us_invalid_date_order():
    assert dcfc_30_360_us(
        start=datetime.date(2020, 2, 1),
        asof=datetime.date(2020, 1, 1),
        end=datetime.date(2020, 3, 1)
    ) == Decimal('0')

    assert dcfc_30_360_us(
        start=datetime.date(2020, 1, 1),
        asof=datetime.date(2020, 3, 1),
        end=datetime.date(2020, 2, 1)
    ) == Decimal('0')


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_26():
    start = datetime.date(2007, 10, 31)
    assert start.day == 31


# LLM-generated content at query #15
#--------------------------

```python
def test_last_payment_date_annual_same_year():
    assert _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)

def test_last_payment_date_annual_previous_year():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)

def test_last_payment_date_semi_annual_july():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_august():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2) == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_april():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2) == datetime.date(2015, 1, 1)

def test_last_payment_date_semi_annual_june_start():
    assert _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2014, 6, 1)

def test_last_payment_date_quarterly():
    assert _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4) == datetime.date(2015, 7, 7)

def test_last_payment_date_december_start():
    assert _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1) == datetime.date(2014, 12, 9)

def test_last_payment_date_december_start_january_end():
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2) == datetime.date(2015, 12, 15)

def test_last_payment_date_december_start_december_end():
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 12, 15)


# LLM-generated content at query #16
#--------------------------

```python
def test_coupon_basic_case():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("25.00"), "USD")

def test_coupon_with_eom():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 6, 15)
    end = datetime.date(2020, 12, 31)
    freq = 2
    eom = 15
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("25.00"), "USD")

def test_coupon_asof_before_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 6, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("0.00"), "USD")

def test_coupon_asof_after_end():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("0.00"), "USD")

def test_coupon_zero_rate():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    principal = Money(1000, "USD")
    rate = Decimal("0.00")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("0.00"), "USD")

def test_coupon_zero_principal():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    principal = Money(0, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("0.00"), "USD")

def test_coupon_high_frequency():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 2, 1)
    end = datetime.date(2020, 12, 31)
    freq = 12
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("4.17"), "USD")

def test_coupon_low_frequency():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 1
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(Decimal("25.00"), "USD")


# LLM-generated content at query #17
#--------------------------

```python
def test_calculate_daily_fraction_with_valid_dates():
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(1/2)

def test_calculate_daily_fraction_with_asof_before_start():
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    start = Date(2023, 1, 1)
    asof = Date(2022, 12, 31)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(0)

def test_calculate_daily_fraction_with_asof_after_end():
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 4)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(0)

def test_calculate_daily_fraction_with_equal_dates():
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 1)
    end = Date(2023, 1, 1)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(0)


# LLM-generated content at query #18
#--------------------------

```python
def test_DCCRegistryMachinery_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #19
#--------------------------

```python
def test_register_raises_typeerror_for_duplicate_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test1", altnames={"ALT1"}, currencies=set(), calculate_fraction_method=lambda *args: ZERO)
    dcc2 = DCC(name="Test2", altnames={"ALT1"}, currencies=set(), calculate_fraction_method=lambda *args: ZERO)
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #20
#--------------------------

```python
def test_find_existing_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("TestDCC", ["AltTestDCC"])
    registry.register(dcc)
    assert registry.find("TestDCC") == dcc

def test_find_existing_alternative_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("TestDCC", ["AltTestDCC"])
    registry.register(dcc)
    assert registry.find("AltTestDCC") == dcc

def test_find_non_existing_name():
    registry = DCCRegistryMachinery()
    assert registry.find("NonExistingDCC") is None

def test_find_case_insensitive():
    registry = DCCRegistryMachinery()
    dcc = DCC("TestDCC", ["AltTestDCC"])
    registry.register(dcc)
    assert registry.find("testdcc") == dcc
    assert registry.find("ALTTESTDCC") == dcc

def test_find_whitespace_insensitive():
    registry = DCCRegistryMachinery()
    dcc = DCC("TestDCC", ["AltTestDCC"])
    registry.register(dcc)
    assert registry.find("  TestDCC  ") == dcc
    assert registry.find("  AltTestDCC  ") == dcc


# LLM-generated content at query #21
#--------------------------

```python
def test_dcfc_act_act_icma():
    ex1_start, ex1_asof, ex1_end = datetime.date(2019, 3, 2), datetime.date(2019, 9, 10), datetime.date(2020, 3, 2)
    assert round(dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end), 10) == Decimal('0.5245901639')


# LLM-generated content at query #22
#--------------------------

```python
def test_dcfc_30_360_isda_example_1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_isda_example_2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_isda_example_3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_isda_example_4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #23
#--------------------------

```python
def test_dcfc_30_e_360_start_day_31():
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 1)
    end = datetime.date(2020, 2, 1)
    result = dcfc_30_e_360(start, asof, end)
    assert result == Decimal('1') / Decimal(360)


# LLM-generated content at query #24
#--------------------------

```python
def test_predicate_evaluates_to_false():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = None
    assert not (eom or start.day)


# LLM-generated content at query #25
#--------------------------

```python
def test_d1_not_31():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    assert dcfc_30_360_us(start, asof, asof) == Decimal('0.16666666666667')


# LLM-generated content at query #26
#--------------------------

```python
def test_init_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #27
#--------------------------

```python
def test_has_leap_day_with_leap_year_in_range():
    start = date(2020, 1, 1)
    end = date(2020, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_without_leap_year_in_range():
    start = date(2021, 1, 1)
    end = date(2021, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_with_leap_day_at_start():
    start = date(2020, 2, 29)
    end = date(2020, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_leap_day_at_end():
    start = date(2020, 1, 1)
    end = date(2020, 2, 29)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_multiple_leap_years_in_range():
    start = date(2016, 1, 1)
    end = date(2024, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_no_leap_years_in_range():
    start = date(2021, 1, 1)
    end = date(2023, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_with_single_day_range_on_leap_day():
    start = date(2020, 2, 29)
    end = date(2020, 2, 29)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_single_day_range_not_on_leap_day():
    start = date(2021, 2, 28)
    end = date(2021, 2, 28)
    assert _has_leap_day(start, end) == False


# LLM-generated content at query #28
#--------------------------

```python
def test_init_creates_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #29
#--------------------------

```python
def test_dcfc_30_360_us_d2_not_31():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    assert not (asof.day == 31 and (start.day in {30, 31}))


# LLM-generated content at query #30
#--------------------------

```python
def test_coupon_standard_case():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(25, "USD")

def test_coupon_with_eom():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 6, 15)
    end = datetime.date(2021, 1, 15)
    freq = 2
    eom = 15
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(25, "USD")

def test_coupon_zero_principal():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(0, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(0, "USD")

def test_coupon_zero_rate():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(0, "USD")

def test_coupon_asof_before_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 6, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(0, "USD")

def test_coupon_asof_after_end():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(0, "USD")


# LLM-generated content at query #31
#--------------------------

```python
def test_last_day_of_month():
    assert _is_last_day_of_month(Date(2023, 1, 31)) == True
    assert _is_last_day_of_month(Date(2023, 2, 28)) == True
    assert _is_last_day_of_month(Date(2023, 3, 31)) == True
    assert _is_last_day_of_month(Date(2023, 4, 30)) == True
    assert _is_last_day_of_month(Date(2023, 5, 31)) == True

def test_not_last_day_of_month():
    assert _is_last_day_of_month(Date(2023, 1, 30)) == False
    assert _is_last_day_of_month(Date(2023, 2, 27)) == False
    assert _is_last_day_of_month(Date(2023, 3, 30)) == False
    assert _is_last_day_of_month(Date(2023, 4, 29)) == False
    assert _is_last_day_of_month(Date(2023, 5, 30)) == False

def test_leap_year_february():
    assert _is_last_day_of_month(Date(2024, 2, 29)) == True
    assert _is_last_day_of_month(Date(2024, 2, 28)) == False


# LLM-generated content at query #32
#--------------------------

```python
def test_interest_with_valid_inputs():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    principal = Money(1000, "USD")
    rate = Decimal("0.1")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == Money(50, "USD")

def test_interest_with_end_none():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    principal = Money(1000, "USD")
    rate = Decimal("0.1")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    result = dcc.interest(principal, rate, start, asof)
    assert result == Money(50, "USD")

def test_interest_with_invalid_dates():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    principal = Money(1000, "USD")
    rate = Decimal("0.1")
    start = Date(2023, 12, 31)
    asof = Date(2023, 6, 1)
    end = Date(2023, 1, 1)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == Money(0, "USD")


# LLM-generated content at query #33
#--------------------------

```python
def test_register_raises_typeerror_when_dcc_name_already_registered():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc1)
    dcc2 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    with pytest.raises(TypeError, match="Day count convention 'Test' is already registered"):
        registry.register(dcc2)


# LLM-generated content at query #34
#--------------------------

```python
def test_eom_not_false_or_none():
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 12, 31)
    frequency = 1
    eom = 15
    result = _last_payment_date(start, asof, frequency, eom)
    assert eom is not False and eom is not None


# LLM-generated content at query #35
#--------------------------

```python
def test_dcfc_act_365_a_without_leap_day():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16986301369863')

def test_dcfc_act_365_a_with_leap_day():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.17213114754098')

def test_dcfc_act_365_a_long_period():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08196721311475')

def test_dcfc_act_365_a_cross_year():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.32513661202186')


# LLM-generated content at query #36
#--------------------------

```python
def test_dcfc_act_365_a_without_leap_day():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16986301369863')

def test_dcfc_act_365_a_with_leap_day():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.17213114754098')

def test_dcfc_act_365_a_long_period():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08196721311475')

def test_dcfc_act_365_a_crossing_year_boundary():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_365_a(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.32513661202186')


# LLM-generated content at query #37
#--------------------------

```python
def test_construct_date_valid_date():
    result = _construct_date(2023, 5, 15)
    assert result.year == 2023
    assert result.month == 5
    assert result.day == 15

def test_construct_date_invalid_day():
    result = _construct_date(2023, 2, 30)
    assert result.year == 2023
    assert result.month == 2
    assert result.day == 28

def test_construct_date_invalid_month():
    with pytest.raises(ValueError):
        _construct_date(2023, 13, 15)

def test_construct_date_invalid_year():
    with pytest.raises(ValueError):
        _construct_date(-2023, 5, 15)

def test_construct_date_zero_day():
    with pytest.raises(ValueError):
        _construct_date(2023, 5, 0)

def test_construct_date_zero_month():
    with pytest.raises(ValueError):
        _construct_date(2023, 0, 15)

def test_construct_date_zero_year():
    with pytest.raises(ValueError):
        _construct_date(0, 5, 15)


# LLM-generated content at query #38
#--------------------------

```python
def test_construct_date_predicate_false():
    assert not (year <= 0 or month <= 0 or day <= 0) for year, month, day in [(1, 1, 1), (2023, 5, 15), (9999, 12, 31)]


# LLM-generated content at query #39
#--------------------------

```python
def test_dcfc_30_e_plus_360_predicate():
    start_date = datetime.date(2007, 10, 31)
    assert start_date.day == 31


# LLM-generated content at query #40
#--------------------------

```python
def test_last_payment_date_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1) == datetime.date(2014, 12, 9)

def test_last_payment_date_semi_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 7, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2) == datetime.date(2015, 7, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2) == datetime.date(2015, 12, 15)
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 12, 15)

def test_last_payment_date_quarterly_frequency():
    assert _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4) == datetime.date(2015, 7, 7)

def test_last_payment_date_start_after_asof():
    assert _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2014, 6, 1)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_30_360_isda_ex1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_isda(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_isda_ex2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_isda(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_isda_ex3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_isda(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_isda_ex4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_isda(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #2
#--------------------------

```python
def test_dcfc_act_act_icma():
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = round(dcfc_act_act_icma(start=start, asof=asof, end=end), 10)
    assert result == Decimal('0.5245901639')


# LLM-generated content at query #3
#--------------------------

```python
def test_last_payment_date_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)

def test_last_payment_date_annual_frequency_same_year():
    assert _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)

def test_last_payment_date_semi_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_frequency_august():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2) == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_frequency_april():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2) == datetime.date(2015, 1, 1)

def test_last_payment_date_semi_annual_frequency_june_start():
    assert _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2014, 6, 1)

def test_last_payment_date_quarterly_frequency():
    assert _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4) == datetime.date(2015, 7, 7)

def test_last_payment_date_december_start():
    assert _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1) == datetime.date(2014, 12, 9)

def test_last_payment_date_december_start_january_end():
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2) == datetime.date(2015, 12, 15)

def test_last_payment_date_december_start_december_end():
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 12, 15)


# LLM-generated content at query #4
#--------------------------

```python
def test_register_successful():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Test", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc)
    assert registry._find_strict("Test") == dcc
    assert registry._find_strict("TestAlt") == dcc

def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc1)
    dcc2 = DCC(name="Test", altnames={}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    try:
        registry.register(dcc2)
    except TypeError as e:
        assert str(e) == "Day count convention 'Test' is already registered"

def test_register_duplicate_alt_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc1)
    dcc2 = DCC(name="AnotherTest", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    try:
        registry.register(dcc2)
    except TypeError as e:
        assert str(e) == "Day count convention 'AnotherTest' is already registered"


# LLM-generated content at query #5
#--------------------------

```python
def test_calculate_daily_fraction_with_valid_dates():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(1/2)

def test_calculate_daily_fraction_with_asof_equal_to_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 1)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(1/2)

def test_calculate_daily_fraction_with_asof_equal_to_end():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 3)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(0)

def test_calculate_daily_fraction_with_asof_before_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    start = Date(2023, 1, 2)
    asof = Date(2023, 1, 1)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(0)

def test_calculate_daily_fraction_with_asof_after_end():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / (e - s).days))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 4)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(0)


# LLM-generated content at query #6
#--------------------------

```python
def test_register_successful():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Test", altnames={"TEST1", "TEST2"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc)
    assert registry._find_strict("Test") == dcc
    assert registry._find_strict("TEST1") == dcc
    assert registry._find_strict("TEST2") == dcc

def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.6))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)

def test_register_duplicate_alt_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test1", altnames={"TEST"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC(name="Test2", altnames={"TEST"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.6))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #7
#--------------------------

```python
def test_calculate_daily_fraction_predicate_false():
    dcc = DCC("test", set(), set(), lambda s, a, e, f: Decimal(1))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    asof_minus_1 = asof - datetime.timedelta(days=1)
    assert not (asof_minus_1 < start)


# LLM-generated content at query #8
#--------------------------

```python
def test_dcfc_act_act_basic_calculation():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16942884946478')

def test_dcfc_act_act_leap_year():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.17216108990194')

def test_dcfc_act_act_long_period():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08243131970956')

def test_dcfc_act_act_cross_year():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.32625945055768')

def test_dcfc_act_act_invalid_date_range():
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2022, 12, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


# LLM-generated content at query #9
#--------------------------

```python
def test_last_day_of_february_in_leap_year():
    date = Date(2020, 2, 29)
    assert _is_last_day_of_month(date) is True

def test_last_day_of_february_in_non_leap_year():
    date = Date(2019, 2, 28)
    assert _is_last_day_of_month(date) is True

def test_last_day_of_april():
    date = Date(2023, 4, 30)
    assert _is_last_day_of_month(date) is True

def test_non_last_day_of_month():
    date = Date(2023, 5, 15)
    assert _is_last_day_of_month(date) is False

def test_last_day_of_december():
    date = Date(2023, 12, 31)
    assert _is_last_day_of_month(date) is True


# LLM-generated content at query #10
#--------------------------

```python
def test_calculate_fraction_with_valid_dates():
    dcc = DCC(
        name="Test",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)
    assert dcc.calculate_fraction(start, asof, end) == Decimal("0.5")

def test_calculate_fraction_with_invalid_dates():
    dcc = DCC(
        name="Test",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    start = Date(2023, 12, 31)
    asof = Date(2023, 6, 1)
    end = Date(2023, 1, 1)
    assert dcc.calculate_fraction(start, asof, end) == ZERO

def test_calculate_fraction_with_freq():
    dcc = DCC(
        name="Test",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.25") if f == Decimal("4") else Decimal("0.5")
    )
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)
    freq = Decimal("4")
    assert dcc.calculate_fraction(start, asof, end, freq) == Decimal("0.25")


# LLM-generated content at query #11
#--------------------------

```python
def test_dcc_registry_machinery_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #12
#--------------------------

```python
def test_dcfc_30_360_isda_start_day_31():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    assert dcfc_30_360_isda(start=start, asof=asof, end=end) == Decimal('1.08333333333333')


# LLM-generated content at query #13
#--------------------------

```python
def test_register_raises_typeerror_for_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0))
    dcc2 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0))
    registry.register(dcc1)
    with pytest.raises(TypeError) as excinfo:
        registry.register(dcc2)
    assert "Day count convention 'Test' is already registered" in str(excinfo.value)


# LLM-generated content at query #14
#--------------------------

```python
def test_interest_basic_calculation():
    dcc = DCC(
        name="Test",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.5")
    )
    principal = Money(1000, "USD")
    rate = Decimal("0.10")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == Money(50, "USD")

def test_interest_without_end_date():
    dcc = DCC(
        name="Test",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.25")
    )
    principal = Money(2000, "EUR")
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2023, 3, 31)
    result = dcc.interest(principal, rate, start, asof)
    assert result == Money(25, "EUR")

def test_interest_zero_fraction():
    dcc = DCC(
        name="Test",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0")
    )
    principal = Money(5000, "GBP")
    rate = Decimal("0.08")
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 1)
    end = Date(2023, 12, 31)
    result = dcc.interest(principal, rate, start, asof, end)
    assert result == Money(0, "GBP")

def test_interest_with_frequency():
    dcc = DCC(
        name="Test",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=lambda s, a, e, f: Decimal("0.3") if f else Decimal("0.5")
    )
    principal = Money(10000, "JPY")
    rate = Decimal("0.02")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)
    freq = Decimal("2")
    result = dcc.interest(principal, rate, start, asof, end, freq)
    assert result == Money(60, "JPY")


# LLM-generated content at query #15
#--------------------------

```python
def test_DCCRegistryMachinery_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert hasattr(registry, '_buffer_main')
    assert hasattr(registry, '_buffer_altn')
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_main) == 0
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_DCCRegistryMachinery_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #17
#--------------------------

```python
def test_last_day_of_january():
    date = Date(2023, 1, 31)
    assert _is_last_day_of_month(date) is True

def test_last_day_of_february_leap_year():
    date = Date(2024, 2, 29)
    assert _is_last_day_of_month(date) is True

def test_last_day_of_february_non_leap_year():
    date = Date(2023, 2, 28)
    assert _is_last_day_of_month(date) is True

def test_last_day_of_april():
    date = Date(2023, 4, 30)
    assert _is_last_day_of_month(date) is True

def test_non_last_day():
    date = Date(2023, 3, 15)
    assert _is_last_day_of_month(date) is False


# LLM-generated content at query #18
#--------------------------

```python
def test_find_existing_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act")
    registry.register(dcc)
    assert registry.find("Act/Act") == dcc

def test_find_existing_dcc_with_altname():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", altnames=["Actual/Actual"])
    registry.register(dcc)
    assert registry.find("Actual/Actual") == dcc

def test_find_nonexistent_dcc():
    registry = DCCRegistryMachinery()
    assert registry.find("NonExistent") is None

def test_find_with_stripped_uppercase():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act")
    registry.register(dcc)
    assert registry.find(" act/act ") == dcc


# LLM-generated content at query #19
#--------------------------

```python
def test_dates_order_check():
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2022, 12, 31)
    end = datetime.date(2023, 12, 31)
    assert dcfc_act_act(start, asof, end) == ZERO


# LLM-generated content at query #20
#--------------------------

```python
def test_calculate_daily_fraction_when_asof_minus_1_is_not_less_than_start():
    dcc = DCC("test", set(), set(), lambda s, a, e, f: Decimal("1.0"))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    asof_minus_1 = asof - datetime.timedelta(days=1)
    assert not (asof_minus_1 < start)


# LLM-generated content at query #21
#--------------------------

```python
def test_coupon_basic_case():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.5"))
    principal = Money(1000, "USD")
    rate = Decimal("0.1")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(50, "USD")

def test_coupon_with_eom():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.5"))
    principal = Money(1000, "USD")
    rate = Decimal("0.1")
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 6, 15)
    end = datetime.date(2020, 12, 31)
    freq = 2
    eom = 15
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(50, "USD")

def test_coupon_zero_principal():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.5"))
    principal = Money(0, "USD")
    rate = Decimal("0.1")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(0, "USD")

def test_coupon_zero_rate():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.5"))
    principal = Money(1000, "USD")
    rate = Decimal("0")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(0, "USD")

def test_coupon_full_period():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("1.0"))
    principal = Money(1000, "USD")
    rate = Decimal("0.1")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    freq = 1
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(100, "USD")


# LLM-generated content at query #22
#--------------------------

```python
def test_dcc_registry_machinery_initialization():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #23
#--------------------------

```python
def test_calculate_fraction_returns_zero_when_dates_are_invalid():
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(1))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)

    assert dcc.calculate_fraction(start, asof, end) == ZERO


# LLM-generated content at query #24
#--------------------------

```python
def test_dcfc_nl_365_basic_cases():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08219178082192')
    assert round(dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32602739726027')


# LLM-generated content at query #25
#--------------------------

```python
def test_dcc_registry_machinery_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_main) == 0
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #26
#--------------------------

```python
def test_dcc_registry_machinery_initialization():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)


# LLM-generated content at query #27
#--------------------------

```python
def test_last_payment_date_predicate():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)


# LLM-generated content at query #28
#--------------------------

```python
def test_last_payment_date_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 1) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2015, 1, 1)

def test_last_payment_date_semi_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 7, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2) == datetime.date(2015, 7, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2014, 6, 1)

def test_last_payment_date_quarterly_frequency():
    assert _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4) == datetime.date(2015, 7, 7)

def test_last_payment_date_with_eom():
    assert _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1) == datetime.date(2014, 12, 9)
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2) == datetime.date(2015, 12, 15)
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 12, 15)


# LLM-generated content at query #29
#--------------------------

```python
def test_initialization_of_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #30
#--------------------------

```python
def test_last_day_of_february_in_leap_year():
    date = Date(2020, 2, 29)
    assert _is_last_day_of_month(date) is True

def test_last_day_of_february_in_non_leap_year():
    date = Date(2021, 2, 28)
    assert _is_last_day_of_month(date) is True

def test_last_day_of_april():
    date = Date(2021, 4, 30)
    assert _is_last_day_of_month(date) is True

def test_last_day_of_december():
    date = Date(2021, 12, 31)
    assert _is_last_day_of_month(date) is True

def test_not_last_day_of_month():
    date = Date(2021, 5, 15)
    assert _is_last_day_of_month(date) is False


# LLM-generated content at query #31
#--------------------------

```python
def test_dcfc_30_360_isda_predicate_false():
    start = datetime.date(2023, 1, 15)
    asof = datetime.date(2023, 2, 1)
    end = datetime.date(2023, 3, 1)
    assert dcfc_30_360_isda(start, asof, end) == Decimal('0.1666666666666666666666666667')


# LLM-generated content at query #32
#--------------------------

```python
def test_dcfc_act_act_icma_returns_correct_fraction():
    ex1_start, ex1_asof, ex1_end = datetime.date(2019, 3, 2), datetime.date(2019, 9, 10), datetime.date(2020, 3, 2)
    assert round(dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end), 10) == Decimal('0.5245901639')


# LLM-generated content at query #33
#--------------------------

```python
def test_init_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert hasattr(registry, '_buffer_main')
    assert hasattr(registry, '_buffer_altn')
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #34
#--------------------------

```python
def test_init_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)


# LLM-generated content at query #35
#--------------------------

```python
def test_dcfc_act_act_basic():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_act_act(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('0.16942884946478')

def test_dcfc_act_act_leap_year():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_act_act(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('0.17216108990194')

def test_dcfc_act_act_long_period():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_act_act(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('1.08243131970956')

def test_dcfc_act_act_cross_year():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_act_act(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('1.32625945055768')


# LLM-generated content at query #36
#--------------------------

```python
def test_init_creates_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #37
#--------------------------

```python
def test_interest_uses_calculate_fraction():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("100"), Currency("USD"))
    rate = Decimal("0.1")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)
    freq = Decimal("1")

    result = dcc.interest(principal, rate, start, asof, end, freq)
    expected = principal * rate * Decimal("0.5")

    assert result == expected


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_evaluates_to_false():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = 1
    assert not (eom or start.day)


# LLM-generated content at query #39
#--------------------------

```python
def test_register_successful():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Test", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc)
    assert registry._buffer_main["Test"] is dcc
    assert registry._buffer_altn["TestAlt"] is dcc

def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc1)
    dcc2 = DCC(name="Test", altnames={"AnotherAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.6))
    with pytest.raises(TypeError, match="Day count convention 'Test' is already registered"):
        registry.register(dcc2)

def test_register_duplicate_alternative_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc1)
    dcc2 = DCC(name="Another", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.6))
    with pytest.raises(TypeError, match="Day count convention 'Another' is already registered"):
        registry.register(dcc2)


# LLM-generated content at query #40
#--------------------------

```python
def test_calculate_daily_fraction_asof_minus_1_less_than_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal(1))
    start = Date(2023, 1, 2)
    asof = Date(2023, 1, 1)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    assert result == Decimal(1)


# LLM-generated content at query #41
#--------------------------

```python
def test_last_day_of_month():
    assert _is_last_day_of_month(Date(2023, 1, 31)) == True
    assert _is_last_day_of_month(Date(2023, 2, 28)) == True
    assert _is_last_day_of_month(Date(2023, 3, 31)) == True
    assert _is_last_day_of_month(Date(2023, 4, 30)) == True
    assert _is_last_day_of_month(Date(2023, 5, 31)) == True
    assert _is_last_day_of_month(Date(2023, 6, 30)) == True
    assert _is_last_day_of_month(Date(2023, 7, 31)) == True
    assert _is_last_day_of_month(Date(2023, 8, 31)) == True
    assert _is_last_day_of_month(Date(2023, 9, 30)) == True
    assert _is_last_day_of_month(Date(2023, 10, 31)) == True
    assert _is_last_day_of_month(Date(2023, 11, 30)) == True
    assert _is_last_day_of_month(Date(2023, 12, 31)) == True

def test_not_last_day_of_month():
    assert _is_last_day_of_month(Date(2023, 1, 30)) == False
    assert _is_last_day_of_month(Date(2023, 2, 27)) == False
    assert _is_last_day_of_month(Date(2023, 3, 30)) == False
    assert _is_last_day_of_month(Date(2023, 4, 29)) == False
    assert _is_last_day_of_month(Date(2023, 5, 30)) == False
    assert _is_last_day_of_month(Date(2023, 6, 29)) == False
    assert _is_last_day_of_month(Date(2023, 7, 30)) == False
    assert _is_last_day_of_month(Date(2023, 8, 30)) == False
    assert _is_last_day_of_month(Date(2023, 9, 29)) == False
    assert _is_last_day_of_month(Date(2023, 10, 30)) == False
    assert _is_last_day_of_month(Date(2023, 11, 29)) == False
    assert _is_last_day_of_month(Date(2023, 12, 30)) == False

def test_leap_year_february():
    assert _is_last_day_of_month(Date(2024, 2, 29)) == True
    assert _is_last_day_of_month(Date(2024, 2, 28)) == False


