####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_is_last_day_of_month_for_january_31st():
    date = Date(2023, 1, 31)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_for_february_28th_non_leap_year():
    date = Date(2023, 2, 28)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_for_february_29th_leap_year():
    date = Date(2024, 2, 29)
    assert _is_last_day_of_month(date) == True

def test_is_last_day_of_month_for_april_30th():
    date = Date(2023, 4, 30)
    assert _is_last_day_of_month(date) == True

def test_is_not_last_day_of_month_for_january_30th():
    date = Date(2023, 1, 30)
    assert _is_last_day_of_month(date) == False

def test_is_not_last_day_of_month_for_february_27th_non_leap_year():
    date = Date(2023, 2, 27)
    assert _is_last_day_of_month(date) == False

def test_is_not_last_day_of_month_for_april_29th():
    date = Date(2023, 4, 29)
    assert _is_last_day_of_month(date) == False


# LLM-generated content at query #2
#--------------------------

```python
def test_register_successful_registration():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Test", altnames={"TEST"}, currencies=set(), calculate_fraction_method=lambda *args: Decimal(0.5))
    registry.register(dcc)
    assert registry._find_strict("Test") == dcc
    assert registry._find_strict("TEST") == dcc

def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda *args: Decimal(0.5))
    dcc2 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda *args: Decimal(0.5))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)

def test_register_duplicate_alternative_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test1", altnames={"TEST2"}, currencies=set(), calculate_fraction_method=lambda *args: Decimal(0.5))
    dcc2 = DCC(name="Test2", altnames=set(), currencies=set(), calculate_fraction_method=lambda *args: Decimal(0.5))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)

def test_register_alternative_name_conflict_with_main():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test1", altnames=set(), currencies=set(), calculate_fraction_method=lambda *args: Decimal(0.5))
    dcc2 = DCC(name="Test2", altnames={"TEST1"}, currencies=set(), calculate_fraction_method=lambda *args: Decimal(0.5))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #3
#--------------------------

```python
def test_dcfc_nl_365():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08219178082192')
    assert round(dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32602739726027')


# LLM-generated content at query #4
#--------------------------

```python
def test_find_existing_dcc_by_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: Decimal('0.5'))
    registry.register(dcc)
    assert registry.find("Act/Act") == dcc

def test_find_existing_dcc_by_alternative_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: Decimal('0.5'))
    registry.register(dcc)
    assert registry.find("ACT/ACT") == dcc

def test_find_existing_dcc_case_insensitive():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: Decimal('0.5'))
    registry.register(dcc)
    assert registry.find("act/act") == dcc

def test_find_existing_dcc_with_whitespace():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: Decimal('0.5'))
    registry.register(dcc)
    assert registry.find("  Act/Act  ") == dcc

def test_find_nonexistent_dcc():
    registry = DCCRegistryMachinery()
    assert registry.find("NonExistent") is None


# LLM-generated content at query #5
#--------------------------

```python
def test_find_existing_dcc_by_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: 0.5)
    registry.register(dcc)
    assert registry.find("Act/Act") == dcc

def test_find_existing_dcc_by_alternative_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: 0.5)
    registry.register(dcc)
    assert registry.find("ACT/ACT") == dcc

def test_find_non_existing_dcc():
    registry = DCCRegistryMachinery()
    assert registry.find("NonExistent") is None

def test_find_with_whitespace_and_case_insensitivity():
    registry = DCCRegistryMachinery()
    dcc = DCC("Act/Act", ["ACT/ACT"], lambda s, e, r: 0.5)
    registry.register(dcc)
    assert registry.find(" act/act ") == dcc


# LLM-generated content at query #6
#--------------------------

```python
def test_find_returns_correct_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC("Test", ["Test"], lambda s, e, r: 1, lambda p, r, s, e, r: p)
    registry.register(dcc)
    assert registry.find("test") == dcc
    assert registry.find("TEST") == dcc
    assert registry.find(" test ") == dcc


# LLM-generated content at query #7
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
    expected = Money(25, "USD")
    assert result == expected

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
    expected = Money(25, "USD")
    assert result == expected

def test_coupon_zero_principal():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(0, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(0, "USD")
    assert result == expected

def test_coupon_zero_rate():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.0")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(0, "USD")
    assert result == expected

def test_coupon_same_start_and_asof():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal((a - s).days / 360))
    principal = Money(1000, "USD")
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    result = dcc.coupon(principal, rate, start, asof, end, freq)
    expected = Money(0, "USD")
    assert result == expected


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

```python
def test_last_payment_date_annual_january_start():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)

def test_last_payment_date_annual_january_start_same_year():
    assert _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)

def test_last_payment_date_semi_annual_january_start():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_january_start_august_asof():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2) == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_january_start_april_asof():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2) == datetime.date(2015, 1, 1)

def test_last_payment_date_annual_june_start():
    assert _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2014, 6, 1)

def test_last_payment_date_quarterly_july_start():
    assert _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4) == datetime.date(2015, 7, 7)

def test_last_payment_date_annual_december_start():
    assert _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1) == datetime.date(2014, 12, 9)

def test_last_payment_date_semi_annual_december_start():
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2) == datetime.date(2015, 12, 15)

def test_last_payment_date_semi_annual_december_start_same_year():
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 12, 15)


# LLM-generated content at query #10
#--------------------------

```python
def test_register_raises_typeerror_when_dcc_name_already_registered():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda *args: ZERO)
    dcc2 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda *args: ZERO)
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #11
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

def test_has_leap_day_with_leap_day_in_range():
    start = datetime.date(2020, 2, 28)
    end = datetime.date(2020, 3, 1)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_leap_day_not_in_range():
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_with_multiple_leap_years_in_range():
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2024, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_no_leap_years_in_range():
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2023, 12, 31)
    assert _has_leap_day(start, end) == False


# LLM-generated content at query #12
#--------------------------

```python
def test_interest_basic_case():
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
    expected = Money(1000, Currency("USD")) * Decimal("0.05") * (Decimal(151) / Decimal(360))
    assert result == expected

def test_interest_without_end_date():
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
    expected = Money(1000, Currency("USD")) * Decimal("0.05") * (Decimal(151) / Decimal(360))
    assert result == expected

def test_interest_zero_fraction():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(360)
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2022, 12, 31)
    end = Date(2023, 12, 31)
    result = dcc.interest(principal, rate, start, asof, end)
    expected = Money(0, Currency("USD"))
    assert result == expected

def test_interest_with_frequency():
    dcc = DCC(
        name="ACT/360",
        altnames={"ACT/360"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal((a - s).days) / Decimal(360) if f is None else Decimal((a - s).days) / (Decimal(360) * f)
    )
    principal = Money(1000, Currency("USD"))
    rate = Decimal("0.05")
    start = Date(2023, 1, 1)
    asof = Date(2023, 6, 1)
    end = Date(2023, 12, 31)
    freq = Decimal("2")
    result = dcc.interest(principal, rate, start, asof, end, freq)
    expected = Money(1000, Currency("USD")) * Decimal("0.05") * (Decimal(151) / (Decimal(360) * Decimal("2")))
    assert result == expected


# LLM-generated content at query #13
#--------------------------

```python
def test_find_returns_correct_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC("Test", ["TestAlt"])
    registry.register(dcc)
    assert registry.find("Test") == dcc
    assert registry.find("test") == dcc
    assert registry.find("TEST") == dcc
    assert registry.find(" Test ") == dcc
    assert registry.find("TestAlt") == dcc
    assert registry.find("testalt") == dcc
    assert registry.find("TESTALT") == dcc
    assert registry.find(" TestAlt ") == dcc


# LLM-generated content at query #14
#--------------------------

```python
def test_dcfc_30_360_us_basic_cases():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)

    assert round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')


# LLM-generated content at query #15
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
    assert list(_get_date_range(start, end)) == [
        Date(2023, 1, 1),
        Date(2023, 1, 2),
        Date(2023, 1, 3),
        Date(2023, 1, 4)
    ]

def test_year_boundary():
    start = Date(2022, 12, 30)
    end = Date(2023, 1, 3)
    assert list(_get_date_range(start, end)) == [
        Date(2022, 12, 30),
        Date(2022, 12, 31),
        Date(2023, 1, 1),
        Date(2023, 1, 2)
    ]


# LLM-generated content at query #16
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


# LLM-generated content at query #17
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


# LLM-generated content at query #18
#--------------------------

```python
def test_dcfc_act_act_icma():
    ex1_start, ex1_asof, ex1_end = datetime.date(2019, 3, 2), datetime.date(2019, 9, 10), datetime.date(2020, 3, 2)
    assert round(dcfc_act_act_icma(start=ex1_start, asof=ex1_asof, end=ex1_end), 10) == Decimal('0.5245901639')


# LLM-generated content at query #19
#--------------------------

```python
def test_register_raises_typeerror_for_duplicate_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test1", altnames={"ALT1"}, currencies=set(), calculate_fraction_method=lambda *args: Decimal(0))
    dcc2 = DCC(name="Test2", altnames={"ALT1"}, currencies=set(), calculate_fraction_method=lambda *args: Decimal(0))
    registry.register(dcc1)
    with pytest.raises(TypeError, match="Day count convention 'Test2' is already registered"):
        registry.register(dcc2)


# LLM-generated content at query #20
#--------------------------

```python
def test_dcfc_act_act_basic():
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


# LLM-generated content at query #21
#--------------------------

```python
def test_dcfc_act_act_predicate_false():
    start = datetime.date(2008, 2, 29)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    assert not (start <= asof <= end)


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_false():
    assert not (False)


# LLM-generated content at query #23
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


# LLM-generated content at query #24
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


# LLM-generated content at query #25
#--------------------------

```python
def test_register_altname_conflict():
    registry = DCCRegistryMachinery()
    dcc1 = DCC("Test1", {"Test1"}, set(), lambda s, a, e, f: Decimal(0))
    dcc2 = DCC("Test2", {"Test1"}, set(), lambda s, a, e, f: Decimal(0))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #26
#--------------------------

```python
def test_dcfc_act_act_predicate():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    date_range = _get_date_range(start, asof)
    years = {year: calendar.isleap(year) for year in range(start.year, asof.year + 1)}
    buffer = [0, 0]
    for date in date_range:
        if years[date.year]:
            buffer[1] += 1
        else:
            buffer[0] += 1
    assert buffer[0] == 32 and buffer[1] == 1


# LLM-generated content at query #27
#--------------------------

```python
def test_init_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)


# LLM-generated content at query #28
#--------------------------

```python
def test_dcfc_30_e_plus_360_example1():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

def test_dcfc_30_e_plus_360_example2():
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

def test_dcfc_30_e_plus_360_example3():
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

def test_dcfc_30_e_plus_360_example4():
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')

def test_dcfc_30_e_plus_360_start_day_31():
    start = datetime.date(2007, 12, 31)
    asof = datetime.date(2008, 1, 31)
    assert round(dcfc_30_e_plus_360(start=start, asof=asof, end=asof), 14) == Decimal('0.02777777777778')

def test_dcfc_30_e_plus_360_asof_day_31():
    start = datetime.date(2007, 12, 30)
    asof = datetime.date(2008, 1, 31)
    expected_asof = datetime.date(2008, 2, 1)
    assert round(dcfc_30_e_plus_360(start=start, asof=asof, end=asof), 14) == round(dcfc_30_e_plus_360(start=start, asof=expected_asof, end=expected_asof), 14)


# LLM-generated content at query #29
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


# LLM-generated content at query #30
#--------------------------

```python
def test_init_initializes_main_buffer():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert len(registry._buffer_main) == 0


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_dcfc_nl_365():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16986301369863')
    assert round(dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08219178082192')
    assert round(dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.32602739726027')


# LLM-generated content at query #2
#--------------------------

```python
def test_register_successful():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Test", altnames={"TestAlt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc)
    assert registry._find_strict("Test") == dcc
    assert registry._find_strict("TestAlt") == dcc

def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)

def test_register_duplicate_alternative_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test1", altnames={"TestAlt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    dcc2 = DCC(name="Test2", altnames={"TestAlt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0.5))
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #3
#--------------------------

```python
def test_dcfc_30_360_us_basic_cases():
    assert round(dcfc_30_360_us(start=datetime.date(2007, 12, 28), asof=datetime.date(2008, 2, 28), end=datetime.date(2008, 2, 28)), 14) == Decimal('0.16666666666667')
    assert round(dcfc_30_360_us(start=datetime.date(2007, 12, 28), asof=datetime.date(2008, 2, 29), end=datetime.date(2008, 2, 29)), 14) == Decimal('0.16944444444444')
    assert round(dcfc_30_360_us(start=datetime.date(2007, 10, 31), asof=datetime.date(2008, 11, 30), end=datetime.date(2008, 11, 30)), 14) == Decimal('1.08333333333333')
    assert round(dcfc_30_360_us(start=datetime.date(2008, 2, 1), asof=datetime.date(2009, 5, 31), end=datetime.date(2009, 5, 31)), 14) == Decimal('1.33333333333333')

def test_dcfc_30_360_us_last_day_of_month():
    assert round(dcfc_30_360_us(start=datetime.date(2023, 1, 31), asof=datetime.date(2023, 2, 28), end=datetime.date(2023, 2, 28)), 14) == Decimal('0.0')
    assert round(dcfc_30_360_us(start=datetime.date(2023, 1, 31), asof=datetime.date(2023, 3, 31), end=datetime.date(2023, 3, 31)), 14) == Decimal('0.16666666666667')

def test_dcfc_30_360_us_d2_adjustment():
    assert round(dcfc_30_360_us(start=datetime.date(2023, 1, 30), asof=datetime.date(2023, 2, 28), end=datetime.date(2023, 2, 28)), 14) == Decimal('0.0')
    assert round(dcfc_30_360_us(start=datetime.date(2023, 1, 31), asof=datetime.date(2023, 2, 28), end=datetime.date(2023, 2, 28)), 14) == Decimal('0.0')

def test_dcfc_30_360_us_d1_adjustment():
    assert round(dcfc_30_360_us(start=datetime.date(2023, 1, 31), asof=datetime.date(2023, 2, 1), end=datetime.date(2023, 2, 1)), 14) == Decimal('0.0')
    assert round(dcfc_30_360_us(start=datetime.date(2023, 1, 31), asof=datetime.date(2023, 2, 2), end=datetime.date(2023, 2, 2)), 14) == Decimal('0.00277777777778')

def test_dcfc_30_360_us_invalid_date_range():
    assert dcfc_30_360_us(start=datetime.date(2023, 2, 1), asof=datetime.date(2023, 1, 31), end=datetime.date(2023, 2, 1)) == Decimal('0')
    assert dcfc_30_360_us(start=datetime.date(2023, 2, 1), asof=datetime.date(2023, 2, 1), end=datetime.date(2023, 1, 31)) == Decimal('0')


# LLM-generated content at query #4
#--------------------------

```python
def test_find_existing_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Test", ["Alt1", "Alt2"])
    registry.register(dcc)
    assert registry.find("Test") == dcc

def test_find_existing_alternative_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Test", ["Alt1", "Alt2"])
    registry.register(dcc)
    assert registry.find("Alt1") == dcc

def test_find_nonexistent_name():
    registry = DCCRegistryMachinery()
    assert registry.find("Nonexistent") is None

def test_find_stripped_uppercase_name():
    registry = DCCRegistryMachinery()
    dcc = DCC("Test", ["Alt1", "Alt2"])
    registry.register(dcc)
    assert registry.find(" test ") == dcc
    assert registry.find("alt1") == dcc


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
def test_empty_range():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 1)
    result = list(_get_date_range(start, end))
    assert result == []

def test_single_day_range():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 2)
    result = list(_get_date_range(start, end))
    assert result == [Date(2023, 1, 1)]

def test_multi_day_range():
    start = Date(2023, 1, 1)
    end = Date(2023, 1, 5)
    result = list(_get_date_range(start, end))
    assert result == [Date(2023, 1, 1), Date(2023, 1, 2), Date(2023, 1, 3), Date(2023, 1, 4)]

def test_year_boundary():
    start = Date(2022, 12, 30)
    end = Date(2023, 1, 3)
    result = list(_get_date_range(start, end))
    assert result == [Date(2022, 12, 30), Date(2022, 12, 31), Date(2023, 1, 1), Date(2023, 1, 2)]

def test_leap_year():
    start = Date(2020, 2, 27)
    end = Date(2020, 3, 2)
    result = list(_get_date_range(start, end))
    assert result == [Date(2020, 2, 27), Date(2020, 2, 28), Date(2020, 2, 29), Date(2020, 3, 1)]


# LLM-generated content at query #7
#--------------------------

```python
def test_last_day_of_month():
    from datetime import date
    assert _is_last_day_of_month(date(2023, 1, 31)) == True
    assert _is_last_day_of_month(date(2023, 2, 28)) == True
    assert _is_last_day_of_month(date(2023, 3, 31)) == True
    assert _is_last_day_of_month(date(2023, 4, 30)) == True
    assert _is_last_day_of_month(date(2023, 5, 31)) == True
    assert _is_last_day_of_month(date(2023, 6, 30)) == True
    assert _is_last_day_of_month(date(2023, 7, 31)) == True
    assert _is_last_day_of_month(date(2023, 8, 31)) == True
    assert _is_last_day_of_month(date(2023, 9, 30)) == True
    assert _is_last_day_of_month(date(2023, 10, 31)) == True
    assert _is_last_day_of_month(date(2023, 11, 30)) == True
    assert _is_last_day_of_month(date(2023, 12, 31)) == True

def test_not_last_day_of_month():
    from datetime import date
    assert _is_last_day_of_month(date(2023, 1, 30)) == False
    assert _is_last_day_of_month(date(2023, 2, 27)) == False
    assert _is_last_day_of_month(date(2023, 3, 30)) == False
    assert _is_last_day_of_month(date(2023, 4, 29)) == False
    assert _is_last_day_of_month(date(2023, 5, 30)) == False
    assert _is_last_day_of_month(date(2023, 6, 29)) == False
    assert _is_last_day_of_month(date(2023, 7, 30)) == False
    assert _is_last_day_of_month(date(2023, 8, 30)) == False
    assert _is_last_day_of_month(date(2023, 9, 29)) == False
    assert _is_last_day_of_month(date(2023, 10, 30)) == False
    assert _is_last_day_of_month(date(2023, 11, 29)) == False
    assert _is_last_day_of_month(date(2023, 12, 30)) == False

def test_leap_year_february():
    from datetime import date
    assert _is_last_day_of_month(date(2020, 2, 29)) == True
    assert _is_last_day_of_month(date(2020, 2, 28)) == False


# LLM-generated content at query #8
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


# LLM-generated content at query #9
#--------------------------

```python
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert isinstance(registry._buffer_main, dict)
    assert isinstance(registry._buffer_altn, dict)
    assert len(registry._buffer_main) == 0
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #10
#--------------------------

```python
def test_register_raises_typeerror_when_dcc_name_already_registered():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Test", altnames={"Test1"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0))
    registry.register(dcc1)
    dcc2 = DCC(name="Test", altnames={"Test2"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal(0))
    with pytest.raises(TypeError, match="Day count convention 'Test' is already registered"):
        registry.register(dcc2)


# LLM-generated content at query #11
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
def test_dcc_registry_machinery_constructor():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #14
#--------------------------

```python
def test_eom_assignment():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = None
    result = eom or start.day
    assert result == start.day


# LLM-generated content at query #15
#--------------------------

```python
def test_DCCRegistryMachinery_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #16
#--------------------------

```python
def test_register_success():
    registry = DCCRegistryMachinery()
    dcc = DCC(
        name="Test",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal(0.5)
    )
    registry.register(dcc)
    assert registry._find_strict("Test") == dcc
    assert registry._find_strict("TestAlt") == dcc

def test_register_duplicate_main_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test",
        altnames={"TestAlt1"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal(0.5)
    )
    dcc2 = DCC(
        name="Test",
        altnames={"TestAlt2"},
        currencies={Currency("EUR")},
        calculate_fraction_method=lambda s, a, e, f: Decimal(0.6)
    )
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)

def test_register_duplicate_alt_name():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(
        name="Test1",
        altnames={"TestAlt"},
        currencies={Currency("USD")},
        calculate_fraction_method=lambda s, a, e, f: Decimal(0.5)
    )
    dcc2 = DCC(
        name="Test2",
        altnames={"TestAlt"},
        currencies={Currency("EUR")},
        calculate_fraction_method=lambda s, a, e, f: Decimal(0.6)
    )
    registry.register(dcc1)
    with pytest.raises(TypeError):
        registry.register(dcc2)


# LLM-generated content at query #17
#--------------------------

```python
def test__is_last_day_of_month__true_for_jan_31():
    assert _is_last_day_of_month(Date(2023, 1, 31)) is True

def test__is_last_day_of_month__false_for_jan_30():
    assert _is_last_day_of_month(Date(2023, 1, 30)) is False

def test__is_last_day_of_month__true_for_feb_28_non_leap_year():
    assert _is_last_day_of_month(Date(2023, 2, 28)) is True

def test__is_last_day_of_month__false_for_feb_27_non_leap_year():
    assert _is_last_day_of_month(Date(2023, 2, 27)) is False

def test__is_last_day_of_month__true_for_feb_29_leap_year():
    assert _is_last_day_of_month(Date(2024, 2, 29)) is True

def test__is_last_day_of_month__false_for_feb_28_leap_year():
    assert _is_last_day_of_month(Date(2024, 2, 28)) is False

def test__is_last_day_of_month__true_for_apr_30():
    assert _is_last_day_of_month(Date(2023, 4, 30)) is True

def test__is_last_day_of_month__false_for_apr_29():
    assert _is_last_day_of_month(Date(2023, 4, 29)) is False

def test__is_last_day_of_month__true_for_dec_31():
    assert _is_last_day_of_month(Date(2023, 12, 31)) is True

def test__is_last_day_of_month__false_for_dec_30():
    assert _is_last_day_of_month(Date(2023, 12, 30)) is False


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_1():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)


# LLM-generated content at query #19
#--------------------------

```python
def test_predicate_evaluates_to_false():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = None
    assert not (eom or start.day)


# LLM-generated content at query #20
#--------------------------

```python
def test_init_initializes_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #21
#--------------------------

```python
def test_dcfc_30_360_us_predicate():
    assert dcfc_30_360_us(
        start=datetime.date(2008, 2, 29),
        asof=datetime.date(2008, 3, 31),
        end=datetime.date(2008, 3, 31)
    ) == Decimal('0.027777777777777777')


# LLM-generated content at query #22
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

def test_dcfc_act_act_another_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.32625945055768')

def test_dcfc_act_act_invalid_date_range():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2019, 12, 31)
    result = dcfc_act_act(start=start, asof=asof, end=asof)
    assert result == ZERO


# LLM-generated content at query #23
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

def test_dcfc_30_360_german_with_year_boundary():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.08333333333333')

def test_dcfc_30_360_german_with_long_period():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('1.33055555555556')

def test_dcfc_30_360_german_with_invalid_date_order():
    start = datetime.date(2008, 2, 28)
    asof = datetime.date(2007, 12, 28)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert result == Decimal('0')

def test_dcfc_30_360_german_with_start_day_31():
    start = datetime.date(2007, 12, 31)
    asof = datetime.date(2008, 1, 31)
    result = dcfc_30_360_german(start=start, asof=asof, end=asof)
    assert round(result, 14) == Decimal('0.02777777777778')

def test_dcfc_30_360_german_with_asof_day_31_and_end_not_asof():
    start = datetime.date(2007, 12, 1)
    asof = datetime.date(2008, 1, 31)
    end = datetime.date(2008, 2, 1)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('0.08333333333333')

def test_dcfc_30_360_german_with_february_last_day_and_end_not_asof():
    start = datetime.date(2007, 2, 1)
    asof = datetime.date(2007, 2, 28)
    end = datetime.date(2007, 3, 1)
    result = dcfc_30_360_german(start=start, asof=asof, end=end)
    assert round(result, 14) == Decimal('0.05555555555556')


# LLM-generated content at query #24
#--------------------------

```python
def test_dcfc_act_365_a_without_leap_day():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 1, 10)
    end = datetime.date(2021, 1, 10)
    assert dcfc_act_365_a(start, asof, end) == Decimal("9") / Decimal("365")

def test_dcfc_act_365_a_with_leap_day():
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 3, 1)
    assert dcfc_act_365_a(start, asof, end) == Decimal("2") / Decimal("366")

def test_dcfc_act_365_a_full_year_no_leap():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 12, 31)
    end = datetime.date(2021, 12, 31)
    assert dcfc_act_365_a(start, asof, end) == Decimal("1")

def test_dcfc_act_365_a_full_year_with_leap():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2020, 12, 31)
    assert dcfc_act_365_a(start, asof, end) == Decimal("1")

def test_dcfc_act_365_a_partial_year_with_leap():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2020, 6, 30)
    assert dcfc_act_365_a(start, asof, end) == Decimal("181") / Decimal("366")

def test_dcfc_act_365_a_partial_year_without_leap():
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2021, 6, 30)
    end = datetime.date(2021, 6, 30)
    assert dcfc_act_365_a(start, asof, end) == Decimal("180") / Decimal("365")


# LLM-generated content at query #25
#--------------------------

```python
def test_has_leap_day_with_leap_day_in_range():
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_without_leap_day_in_range():
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

def test_has_leap_day_with_multiple_leap_days_in_range():
    start = datetime.date(2019, 1, 1)
    end = datetime.date(2021, 12, 31)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_no_leap_days_in_range():
    start = datetime.date(2019, 1, 1)
    end = datetime.date(2019, 12, 31)
    assert _has_leap_day(start, end) == False

def test_has_leap_day_with_same_start_and_end_date():
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    assert _has_leap_day(start, end) == True

def test_has_leap_day_with_same_start_and_end_date_no_leap_day():
    start = datetime.date(2021, 2, 28)
    end = datetime.date(2021, 2, 28)
    assert _has_leap_day(start, end) == False


# LLM-generated content at query #26
#--------------------------

```python
def test_register_raises_typeerror_when_dcc_name_already_registered():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Test", altnames=set(), currencies=set(), calculate_fraction_method=lambda *args: Decimal(0))
    registry.register(dcc)
    with pytest.raises(TypeError):
        registry.register(dcc)


# LLM-generated content at query #27
#--------------------------

```python
def test_last_payment_date_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)

def test_last_payment_date_annual_frequency_same_year():
    assert _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1) == datetime.date(2015, 1, 1)

def test_last_payment_date_semi_annual_frequency():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_frequency_partial_year():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2) == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_frequency_early_year():
    assert _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2) == datetime.date(2015, 1, 1)

def test_last_payment_date_annual_frequency_mid_year_start():
    assert _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1) == datetime.date(2014, 6, 1)

def test_last_payment_date_quarterly_frequency():
    assert _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4) == datetime.date(2015, 7, 7)

def test_last_payment_date_annual_frequency_december_start():
    assert _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1) == datetime.date(2014, 12, 9)

def test_last_payment_date_semi_annual_frequency_december_start():
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2) == datetime.date(2015, 12, 15)

def test_last_payment_date_semi_annual_frequency_december_start_end_year():
    assert _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2) == datetime.date(2015, 12, 15)


# LLM-generated content at query #28
#--------------------------

```python
def test_find_returns_correct_dcc():
    registry = DCCRegistryMachinery()
    dcc = DCC("Test", ["AltTest"])
    registry.register(dcc)
    assert registry.find("Test") == dcc
    assert registry.find("AltTest") == dcc
    assert registry.find("test") == dcc
    assert registry.find(" alttest ") == dcc


# LLM-generated content at query #29
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
    expected_interest = principal * rate * Decimal(151) / Decimal(360)
    assert dcc.interest(principal, rate, start, asof, end) == expected_interest

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
    expected_interest = principal * rate * Decimal(151) / Decimal(360)
    assert dcc.interest(principal, rate, start, asof) == expected_interest

def test_interest_returns_zero_when_asof_is_before_start():
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

def test_interest_returns_zero_when_asof_is_after_end():
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


# LLM-generated content at query #30
#--------------------------

```python
def test_dcfc_act_act_icma():
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    assert round(dcfc_act_act_icma(start=start, asof=asof, end=end), 10) == Decimal('0.5245901639')


