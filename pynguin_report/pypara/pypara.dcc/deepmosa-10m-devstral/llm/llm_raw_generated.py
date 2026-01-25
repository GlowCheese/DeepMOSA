####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_register_success():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Test", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc)
    assert registry._find_strict("Test") == dcc
    assert registry._find_strict("TestAlt") == dcc

def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC(name="Test", altnames={}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.6))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)

def test_register_duplicate_alt_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test1", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC(name="Test2", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.6))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #2
#--------------------------

```python
def test_has_leap_day_with_leap_year_in_range():
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_without_leap_year_in_range():
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_with_leap_day_exactly_at_start():
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_leap_day_exactly_at_end():
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 2, 29)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_multiple_leap_years_in_range():
    start = datetime.date(2016, 1, 1)
    end = datetime.date(2024, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_no_leap_years_in_range():
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2022, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_with_same_start_and_end_date_on_leap_day():
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_same_start_and_end_date_not_on_leap_day():
    start = datetime.date(2021, 2, 28)
    end = datetime.date(2021, 2, 28)
    assert _has_leap_day(start, end) == False


# LLM-generated content at query #3
#--------------------------

```python
def test_last_day_of_february_non_leap_year():
    date = Date(2023, 2, 28)
    assert _is_last_day_of_month(date) == True

def test_last_day_of_february_leap_year():
    date = Date(2024, 2, 29)
    assert _is_last_day_of_month(date) == True

def test_last_day_of_january():
    date = Date(2023, 1, 31)
    assert _is_last_day_of_month(date) == True

def test_last_day_of_april():
    date = Date(2023, 4, 30)
    assert _is_last_day_of_month(date) == True

def test_not_last_day_of_month():
    date = Date(2023, 3, 15)
    assert _is_last_day_of_month(date) == False


# LLM-generated content at query #4
#--------------------------

```python
def test_last_payment_date_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)
    assert _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)

def test_last_payment_date_semi_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 7, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2) == datetime.date(2015, 7, 1)
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2) == datetime.date(2015, 1, 1)

def test_last_payment_date_start_date_before_asof():
    assert _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2014, 6, 1)

def test_last_payment_date_quarterly_frequency():
    assert _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4) == datetime.date(2015, 7, 7)

def test_last_payment_date_same_month_different_year():
    assert _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1) == datetime.date(2014, 12, 9)

def test_last_payment_date_multi_year_period():
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2) == datetime.date(2015, 12, 15)
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 12, 15)


# LLM-generated content at query #5
#--------------------------

```python
def test_DCCRegistryMachinery_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #6
#--------------------------

```python
def test_dcfc_act_365_l():
    assert round(dcfc_act_365_l(start=datetime.date(2007, 12, 28), asof=datetime.date(2008, 2, 28), end=datetime.date(2008, 2, 28)), 14) == Decimal('0.16939890710383')
    assert round(dcfc_act_365_l(start=datetime.date(2007, 12, 28), asof=datetime.date(2008, 2, 29), end=datetime.date(2008, 2, 29)), 14) == Decimal('0.17213114754098')
    assert round(dcfc_act_365_l(start=datetime.date(2007, 10, 31), asof=datetime.date(2008, 11, 30), end=datetime.date(2008, 11, 30)), 14) == Decimal('1.08196721311475')
    assert round(dcfc_act_365_l(start=datetime.date(2008, 2, 1), asof=datetime.date(2009, 5, 31), end=datetime.date(2009, 5, 31)), 14) == Decimal('1.32876712328767')


# LLM-generated content at query #7
#--------------------------

```python
def test_get_date_range_empty():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 1)
    result = list(_get_date_range(start, end))
    assert result == []

def test_get_date_range_single_day():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 2)
    result = list(_get_date_range(start, end))
    assert result == [Date(2023, 1, 1)]

def test_get_date_range_multiple_days():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 5)
    result = list(_get_date_range(start, end))
    assert result == [Date(2023, 1, 1), Date(2023, 1, 2), Date(2023, 1, 3), Date(2023, 1, 4)]

def test_get_date_range_year_boundary():
    start = Date(2022, 12, 30)
    end = Date(2023, 1, 3)
    result = list(_get_date_range(start, end))
    assert result == [Date(2022, 12, 30), Date(2022, 12, 31), Date(2023, 1, 1), Date(2023, 1, 2)]

def test_get_date_range_month_boundary():
    start = Date(2023, 1, 30)
    end = Date(2023, 2, 3)
    result = list(_get_date_range(start, end))
    assert result == [Date(2023, 1, 30), Date(2023, 1, 31), Date(2023, 2, 1), Date(2023, 2, 2)]


# LLM-generated content at query #8
#--------------------------

```python
def test_dcc_registry_machinery_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert hasattr(registry, '_buffer_main')
    assert isinstance(registry._buffer_main, dict)
    assert len(registry._buffer_main) == 0
    assert hasattr(registry, '_buffer_altn')
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #9
#--------------------------

```python
def test_init_creates_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #10
#--------------------------

```python
def test_coupon_basic_case():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal(0.5))
    principal = Money(1000, "USD")
    rate = Decimal("0.10")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 1
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(50, "USD")

def test_coupon_with_eom():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal(0.25))
    principal = Money(2000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 4, 15)
    end = datetime.date(2020, 7, 15)
    freq = 2
    eom = 15
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(50, "USD")

def test_coupon_zero_principal():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal(0.5))
    principal = Money(0, "USD")
    rate = Decimal("0.10")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 1
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(0, "USD")

def test_coupon_zero_rate():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal(0.5))
    principal = Money(1000, "USD")
    rate = Decimal("0.00")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2020, 12, 31)
    freq = 1
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(0, "USD")

def test_coupon_high_frequency():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal(0.1))
    principal = Money(5000, "USD")
    rate = Decimal("0.08")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 2, 1)
    end = datetime.date(2020, 3, 1)
    freq = 12
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(400, "USD")


# LLM-generated content at query #11
#--------------------------

```python
def test_dcfc_30_360_us_basic_cases():
    assert round(dcfc_30_360_us(start=datetime.date(2007, 12, 28), asof=datetime.date(2008, 2, 28), end=datetime.date(2008, 2, 28)), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_us(start=datetime.date(2007, 12, 28), asof=datetime.date(2008, 2, 29), end=datetime.date(2008, 2, 29)), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_us(start=datetime.date(2007, 10, 31), asof=datetime.date(2008, 11, 30), end=datetime.date(2008, 11, 30)), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_us(start=datetime.date(2008, 2, 1), asof=datetime.date(2009, 5, 31), end=datetime.date(2009, 5, 31)), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #12
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

def test_last_payment_date_with_eom():
    assert _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2014, 6, 1)


# LLM-generated content at query #13
#--------------------------

```python
def test_dcfc_30_360_us_basic_cases():
    assert round(dcfc_30_360_us(start=datetime.date(2007, 12, 28), asof=datetime.date(2008, 2, 28), end=datetime.date(2008, 2, 28)), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_us(start=datetime.date(2007, 12, 28), asof=datetime.date(2008, 2, 29), end=datetime.date(2008, 2, 29)), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_us(start=datetime.date(2007, 10, 31), asof=datetime.date(2008, 11, 30), end=datetime.date(2008, 11, 30)), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_us(start=datetime.date(2008, 2, 1), asof=datetime.date(2009, 5, 31), end=datetime.date(2009, 5, 31)), 14) == Decimal('1.33333333333333')

def test_dcfc_30_360_us_edge_cases():
    assert round(dcfc_30_360_us(start=datetime.date(2020, 1, 31), asof=datetime.date(2020, 2, 28), end=datetime.date(2020, 2, 28)), 14) == Decimal('0.0')
    assert round(dcfc_30_360_us(start=datetime.date(2020, 1, 31), asof=datetime.date(2020, 2, 29), end=datetime.date(2020, 2, 29)), 14) == Decimal('0.02777777777778')
    assert round(dcfc_30_360_us(start=datetime.date(2020, 2, 29), asof=datetime.date(2020, 3, 31), end=datetime.date(2020, 3, 31)), 14) == Decimal('0.08333333333333')
    assert round(dcfc_30_360_us(start=datetime.date(2020, 2, 29), asof=datetime.date(2020, 4, 30), end=datetime.date(2020, 4, 30)), 14) == Decimal('0.08333333333333')

def test_dcfc_30_360_us_same_dates():
    assert round(dcfc_30_360_us(start=datetime.date(2020, 1, 1), asof=datetime.date(2020, 1, 1), end=datetime.date(2020, 1, 1)), 14) == Decimal('0.0')
    assert round(dcfc_30_360_us(start=datetime.date(2020, 12, 31), asof=datetime.date(2020, 12, 31), end=datetime.date(2020, 12, 31)), 14) == Decimal('0.0')

def test_dcfc_30_360_us_year_boundaries():
    assert round(dcfc_30_360_us(start=datetime.date(2019, 12, 31), asof=datetime.date(2020, 1, 31), end=datetime.date(2020, 1, 31)), 14) == Decimal('0.0')
    assert round(dcfc_30_360_us(start=datetime.date(2019, 12, 31), asof=datetime.date(2020, 2, 28), end=datetime.date(2020, 2, 28)), 14) == Decimal('0.02777777777778')
    assert round(dcfc_30_360_us(start=datetime.date(2019, 12, 31), asof=datetime.date(2020, 2, 29), end=datetime.date(2020, 2, 29)), 14) == Decimal('0.02777777777778')


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_9():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 1)
    assert (end - start).days == 0


# LLM-generated content at query #15
#--------------------------

```python
def test_has_leap_day_with_leap_year_in_range():
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_without_leap_year_in_range():
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_with_leap_day_at_start():
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_leap_day_at_end():
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 2, 29)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_multiple_leap_years_in_range():
    start = datetime.date(2016, 1, 1)
    end = datetime.date(2024, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_no_leap_years_in_range():
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2023, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_with_single_day_range_leap_day():
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_single_day_range_non_leap_day():
    start = datetime.date(2021, 2, 28)
    end = datetime.date(2021, 2, 28)
    assert _has_leap_day(start, end) == False


# LLM-generated content at query #16
#--------------------------

```python
def test_dcfc_30_360_us_example_1():
    start, asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_us_example_2():
    start, asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_us_example_3():
    start, asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_us_example_4():
    start, asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_us(start=start, asof=asof, end=asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #17
#--------------------------

```python
def test_init_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #18
#--------------------------

```python
def test_register_altname_conflict():
    registry = DCCRegistryMachinery()
    dcc1 = DCC("Test1", {"Test1"}, set(), lambda s, a, e, f: Decimal(1))
    dcc2 = DCC("Test2", {"Test1"}, set(), lambda s, a, e, f: Decimal(1))
    registry.register(dcc1)
    assert registry._is_registered("Test1") is True


# LLM-generated content at query #19
#--------------------------

```python
def test_init_buffer_main_is_dict():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)


# LLM-generated content at query #20
#--------------------------

```python
def test_find_existing_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Test", ["AltTest"])
    registry.register(dcc)
    assert registry.find("Test") == dcc

def test_find_existing_alt_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Test", ["AltTest"])
    registry.register(dcc)
    assert registry.find("AltTest") == dcc

def test_find_nonexistent_name():
    registry = DCCRegistryMachinery()
    assert registry.find("NonExistent") is None

def test_find_stripped_uppercase_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Test", ["AltTest"])
    registry.register(dcc)
    assert registry.find(" test ") == dcc
    assert registry.find("TEST") == dcc
    assert registry.find(" altTest ") == dcc
    assert registry.find("ALTTTEST") == dcc


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
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
    dcc1 = DCC(name="Test", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC(name="Test", altnames={"TestAlt2"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.6))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)

def test_register_duplicate_alternative_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC(name="Test2", altnames={"TestAlt"}, currencies={}, calculate_fraction_method=lambda s, a, e, f: Decimal(0.6))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #3
#--------------------------

```python
def test_last_day_of_month():
    from datetime import date
    import calendar

    assert _is_last_day_of_month(date(2023, 1, 31)) == True
    assert _is_last_day_of_month(date(2023, 2, 28)) == True
    assert _is_last_day_of_month(date(2023, 3, 31)) == True
    assert _is_last_day_of_month(date(2023, 4, 30)) == True
    assert _is_last_day_of_month(date(2023, 5, 31)) == True

def test_not_last_day_of_month():
    from datetime import date

    assert _is_last_day_of_month(date(2023, 1, 30)) == False
    assert _is_last_day_of_month(date(2023, 2, 27)) == False
    assert _is_last_day_of_month(date(2023, 3, 30)) == False
    assert _is_last_day_of_month(date(2023, 4, 29)) == False
    assert _is_last_day_of_month(date(2023, 5, 30)) == False

def test_leap_year_february():
    from datetime import date

    assert _is_last_day_of_month(date(2024, 2, 29)) == True
    assert _is_last_day_of_month(date(2024, 2, 28)) == False


# LLM-generated content at query #4
#--------------------------

```python
def test_dcc_registry_machinery_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert hasattr(registry, '_buffer_main')
    assert isinstance(registry._buffer_main, dict)
    assert len(registry._buffer_main) == 0
    assert hasattr(registry, '_buffer_altn')
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #5
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


# LLM-generated content at query #6
#--------------------------

```python
def test_dcfc_30_360_german_with_standard_dates():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_german_with_leap_year():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.16944444444444')

def test_dcfc_30_360_german_with_year_end():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_german_with_long_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33055555555556')


# LLM-generated content at query #7
#--------------------------

```python
def test_find_existing_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"])
    registry.register(dcc)
    assert registry.find("Act/Act") == dcc

def test_find_existing_alternative_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"])
    registry.register(dcc)
    assert registry.find("ACT/ACT") == dcc

def test_find_non_existing_name():
    registry = DCCRegistryMachinery()
    assert registry.find("NonExistent") is None

def test_find_case_insensitive():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"])
    registry.register(dcc)
    assert registry.find("act/act") == dcc

def test_find_with_whitespace():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"])
    registry.register(dcc)
    assert registry.find("  Act/Act  ") == dcc


# LLM-generated content at query #8
#--------------------------

```python
def test_init_creates_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #9
#--------------------------

```python
def test_register_altname_conflict():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test1", altnames={"Alt1"}, currencies=set(), calculate_fraction_method=lambda *args: ZERO)
    dcc2 = DCC(name="Test2", altnames={"Alt1"}, currencies=set(), calculate_fraction_method=lambda *args: ZERO)
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #10
#--------------------------

```python
def test_coupon_basic_case():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.5"))
    principal = Money(1000, "USD")
    rate = Decimal("0.10")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 1
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(50, "USD")

def test_coupon_with_eom():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.25"))
    principal = Money(1000, "USD")
    rate = Decimal("0.10")
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 4, 15)
    end = datetime.date(2021, 1, 15)
    freq = 4
    eom = 15
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(25, "USD")

def test_coupon_zero_fraction():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0"))
    principal = Money(1000, "USD")
    rate = Decimal("0.10")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = 1
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(0, "USD")

def test_coupon_full_period():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("1"))
    principal = Money(1000, "USD")
    rate = Decimal("0.10")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2021, 1, 1)
    freq = 1
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(100, "USD")

def test_coupon_partial_period():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.75"))
    principal = Money(1000, "USD")
    rate = Decimal("0.10")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 9, 30)
    end = datetime.date(2021, 1, 1)
    freq = 1
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    assert result == Money(75, "USD")


# LLM-generated content at query #11
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
    start = Date(2023, 12, 31)
    asof = Date(2023, 6, 1)
    end = Date(2023, 1, 1)
    assert dcc.calculate_fraction(start, asof, end) == ZERO


# LLM-generated content at query #12
#--------------------------

```python
def test_dcfc_act_365_a():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_act_365_a(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_act_365_a(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.17213114754098')
    assert round(dcfc_act_365_a(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08196721311475')
    assert round(dcfc_act_365_a(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32513661202186')


# LLM-generated content at query #13
#--------------------------

```python
def test_DCCRegistryMachinery_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #14
#--------------------------

```python
def test_dcc_registry_machinery_constructor_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_main) == 0
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #15
#--------------------------

```python
def test_dcfc_act_act():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_act_act(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16942884946478')
    assert round(dcfc_act_act(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.17216108990194')
    assert round(dcfc_act_act(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08243131970956')
    assert round(dcfc_act_act(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32625945055768')


# LLM-generated content at query #16
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

def test_dcfc_30_e_plus_360_start_day_31():
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 1)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('1') / Decimal('360')

def test_dcfc_30_e_plus_360_asof_day_31():
    start = datetime.date(2020, 1, 30)
    asof = datetime.date(2020, 1, 31)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    expected = (datetime.date(2020, 2, 1) - datetime.date(2020, 1, 30)).days / Decimal('360')
    assert result == expected

def test_dcfc_30_e_plus_360_same_day():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    assert dcfc_30_e_plus_360(start=start, asof=asof, end=asof) == Decimal('0')

def test_dcfc_30_e_plus_360_one_year():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    assert dcfc_30_e_plus_360(start=start, asof=asof, end=asof) == Decimal('1')

def test_dcfc_30_e_plus_360_leap_year():
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    result = dcfc_30_e_plus_360(start=start, asof=asof, end=asof)
    assert result == Decimal('2') / Decimal('360')


# LLM-generated content at query #17
#--------------------------

```python
def test_interest_with_valid_dates():
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
    expected_interest = principal * rate * (Decimal(151) / Decimal(360))
    assert dcc.interest(principal, rate, start, asof, end) == expected_interest

def test_interest_with_asof_equal_to_end():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(360)
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2023, 12, 31)
    expected_interest = principal * rate * (Decimal(364) / Decimal(360))
    assert dcc.interest(principal, rate, start, asof) == expected_interest

def test_interest_with_asof_before_start():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(360)
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 6, 1)
    asof = Date(2023, 1, 1)
    end = Date(2023, 12, 31)
    assert dcc.interest(principal, rate, start, asof, end) == Money(0, Currency("USD"))

def test_interest_with_asof_after_end():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(360)
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2024, 6, 1)
    end = Date(2023, 12, 31)
    assert dcc.interest(principal, rate, start, asof, end) == Money(0, Currency("USD"))

def test_interest_with_frequency():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(360) / f if f else Decimal((a - s).days) / Decimal(360)
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)
    freq = Decimal("2")
    expected_interest = principal * rate * (Decimal(151) / Decimal(360) / freq)
    assert dcc.interest(principal, rate, start, asof, end, freq) == expected_interest


# LLM-generated content at query #18
#--------------------------

```python
def test_calculate_fraction_invalid_dates():
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    assert dcc.calculate_fraction(asof, start, end) == ZERO
    assert dcc.calculate_fraction(start, end, asof) == ZERO

def test_calculate_fraction_valid_dates():
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    assert dcc.calculate_fraction(start, asof, end) == Decimal(0.5)

def test_calculate_fraction_with_freq():
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(f) if f else Decimal(0.5))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    assert dcc.calculate_fraction(start, asof, end, Decimal(0.25)) == Decimal(0.25)


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_at_line_1():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)


# LLM-generated content at query #20
#--------------------------

```python
def test_dcfc_act_act_icma():
    ex1_start, ex1_asof, ex1_end = datetime.date(2019, 3, 2), datetime.date(2019, 9, 10), datetime.date(2020, 3, 2)
    assert round(dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end), 10) == Decimal('0.5245901639')

    ex2_start, ex2_asof, ex2_end = datetime.date(2020, 1, 1), datetime.date(2020, 1, 1), datetime.date(2020, 1, 1)
    assert dcfc_act_act_icma(start=ex2_start, asof=ex2_asof, end=ex2_end) == ZERO

    ex3_start, ex3_asof, ex3_end = datetime.date(2020, 1, 1), datetime.date(2020, 1, 2), datetime.date(2020, 1, 3)
    assert dcfc_act_act_icma(start=ex3_start, asof=ex3_asof, end=ex3_end) == Decimal('0.5')


# LLM-generated content at query #21
#--------------------------

```python
def test_dcfc_act_365_a():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_act_365_a(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_act_365_a(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.17213114754098')
    assert round(dcfc_act_365_a(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08196721311475')
    assert round(dcfc_act_365_a(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32513661202186')


# LLM-generated content at query #22
#--------------------------

```python
def test_DCCRegistryMachinery_constructor():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #23
#--------------------------

```python
def test_dcfc_30_360_german_predicate_true():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    assert dcfc_30_360_german(start=start, asof=asof, end=end) == Decimal('0.16944444444444')


# LLM-generated content at query #24
#--------------------------

```python
def test_init_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #25
#--------------------------

```python
def test_coupon_standard_case():
    dcc = DCC("Test", {"Test"}, {"USD"}, lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(25, "USD")
    assert result == expected

def test_coupon_with_eom():
    dcc = DCC("Test", {"Test"}, {"USD"}, lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 6, 15)
    end = datetime.date(2021, 1, 15)
    freq = 2
    eom = 15
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = Money(25, "USD")
    assert result == expected

def test_coupon_asof_before_start():
    dcc = DCC("Test", {"Test"}, {"USD"}, lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2019, 12, 1)
    end = datetime.date(2021, 1, 1)
    freq = 1
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(0, "USD")
    assert result == expected

def test_coupon_asof_after_end():
    dcc = DCC("Test", {"Test"}, {"USD"}, lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2022, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = 1
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(0, "USD")
    assert result == expected

def test_coupon_frequency_4():
    dcc = DCC("Test", {"Test"}, {"USD"}, lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 4, 1)
    end = datetime.date(2021, 1, 1)
    freq = 4
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(12.5, "USD")
    assert result == expected


# LLM-generated content at query #26
#--------------------------

```python
def test_30_e_plus_360_predicate_at_line_30():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    assert not (asof.day == 31)


# LLM-generated content at query #27
#--------------------------

```python
def test_dcc_registry_machinery_initialization():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)


# LLM-generated content at query #28
#--------------------------

```python
def test_DCCRegistryMachinery_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #29
#--------------------------

```python
def test_coupon_basic_case():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(25, "USD")

def test_coupon_with_eom():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
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
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(0, "USD")

def test_coupon_zero_rate():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(0, "USD")

def test_coupon_asof_before_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2019, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(0, "USD")

def test_coupon_asof_after_end():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2022, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    assert result == Money(0, "USD")


# LLM-generated content at query #30
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
    with raises(ValueError):
        _construct_date(2023, 13, 1)

def test_construct_date_invalid_year():
    with raises(ValueError):
        _construct_date(0, 1, 1)

def test_construct_date_negative_values():
    with raises(ValueError):
        _construct_date(-1, -1, -1)

def test_construct_date_leap_year():
    result = _construct_date(2020, 2, 29)
    assert result.year == 2020
    assert result.month == 2
    assert result.day == 29

def test_construct_date_non_leap_year():
    result = _construct_date(2021, 2, 29)
    assert result.year == 2021
    assert result.month == 2
    assert result.day == 28


# LLM-generated content at query #31
#--------------------------

```python
def test_register_raises_typeerror_when_dcc_name_already_registered():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={"TestAlt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc1)
    dcc2 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    with pytest.raises(TypeError, match="Day count convention 'Test' is already registered"):
        registry.register(dcc2)


# LLM-generated content at query #32
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

def test_dcfc_act_act_different_years():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.32625945055768')

def test_dcfc_act_act_invalid_date_range():
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2022, 12, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == Decimal('0')


# LLM-generated content at query #33
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

def test_last_payment_date_invalid_date_handling():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)


# LLM-generated content at query #34
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

def test_last_payment_date_annual_frequency_june_start():
    assert _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2014, 6, 1)

def test_last_payment_date_quarterly_frequency():
    assert _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4) == datetime.date(2015, 7, 7)

def test_last_payment_date_annual_frequency_december_start():
    assert _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1) == datetime.date(2014, 12, 9)

def test_last_payment_date_semi_annual_frequency_january():
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2) == datetime.date(2015, 12, 15)

def test_last_payment_date_semi_annual_frequency_december():
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 12, 15)


