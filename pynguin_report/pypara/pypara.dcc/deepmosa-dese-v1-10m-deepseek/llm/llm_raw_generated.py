####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_register_new_dcc_successfully():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TestAlt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("TestAlt") == dcc

def test_register_duplicate_main_name_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="TestDCC", altnames={"Other"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'TestDCC' is already registered"

def test_register_duplicate_altname_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC1", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="TestDCC2", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'TestDCC2' is already registered"

def test_register_altname_conflict_with_main_name_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC1", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="TestDCC2", altnames={"TestDCC1"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'TestDCC2' is already registered"

def test_register_main_name_conflict_with_altname_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC1", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="AltName", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'AltName' is already registered"

def test_register_multiple_altnames_registers_all():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"Alt1", "Alt2", "Alt3"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("Alt1") == dcc
    assert registry._find_strict("Alt2") == dcc
    assert registry._find_strict("Alt3") == dcc

def test_register_empty_altnames_does_not_add_to_alt_buffer():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._buffer_altn == {}

def test_register_updates_main_buffer():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._buffer_main["TestDCC"] == dcc

def test_register_updates_alt_buffer():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._buffer_altn["AltName"] == dcc


# LLM-generated content at query #2
#--------------------------

def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #3
#--------------------------

def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


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

def test_get_date_range_with_leap_year():
    start = datetime.date(2024, 2, 28)
    end = datetime.date(2024, 3, 2)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2024, 2, 28), datetime.date(2024, 2, 29), datetime.date(2024, 3, 1)]
    assert result == expected

def test_get_date_range_across_month_boundary():
    start = datetime.date(2023, 1, 31)
    end = datetime.date(2023, 2, 3)
    result = list(_get_date_range(start, end))
    expected = [datetime.date(2023, 1, 31), datetime.date(2023, 2, 1), datetime.date(2023, 2, 2)]
    assert result == expected

def test_get_date_range_with_negative_days():
    start = datetime.date(2023, 1, 5)
    end = datetime.date(2023, 1, 2)
    result = list(_get_date_range(start, end))
    expected = []
    assert result == expected


# LLM-generated content at query #5
#--------------------------

def test_register_new_dcc_successfully():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TestAlt"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("TestAlt") == dcc

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
    dcc1 = DCC(name="TestDCC1", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="TestDCC2", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'TestDCC2' is already registered"

def test_register_altname_conflict_with_main_name_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC1", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="TestDCC2", altnames={"TestDCC1"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'TestDCC2' is already registered"

def test_register_main_name_conflict_with_altname_raises_error():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="TestDCC1", altnames={"AltName"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    dcc2 = DCC(name="AltName", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    registry.register(dcc1)
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'AltName' is already registered"

def test_register_multiple_altnames_successfully():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"Alt1", "Alt2", "Alt3"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("Alt1") == dcc
    assert registry._find_strict("Alt2") == dcc
    assert registry._find_strict("Alt3") == dcc

def test_register_empty_altnames_successfully():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc
    assert registry._find_strict("NonExistent") is None


# LLM-generated content at query #6
#--------------------------

def test_has_leap_day_with_leap_day_in_range():
    start = datetime.date(2020, 2, 28)
    end = datetime.date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_without_leap_day_in_range():
    start = datetime.date(2021, 2, 28)
    end = datetime.date(2021, 3, 1)
    result = _has_leap_day(start, end)
    assert result == False

def test_has_leap_day_range_starting_on_leap_day():
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_range_ending_on_leap_day():
    start = datetime.date(2020, 2, 28)
    end = datetime.date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_range_covering_multiple_years_with_leap_day():
    start = datetime.date(2019, 12, 31)
    end = datetime.date(2021, 1, 1)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_range_covering_multiple_years_without_leap_day():
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2023, 12, 31)
    result = _has_leap_day(start, end)
    assert result == False

def test_has_leap_day_single_day_range_on_leap_day():
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_single_day_range_not_on_leap_day():
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result == False


# LLM-generated content at query #7
#--------------------------

def test_coupon_regular_payment_schedule():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("1000"), Currency.USD)
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.5")
    assert result == expected

def test_coupon_with_eom_adjustment():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.25"))
    principal = Money(Decimal("2000"), Currency.USD)
    rate = Decimal("0.03")
    start = datetime.date(2020, 2, 15)
    asof = datetime.date(2020, 5, 15)
    end = datetime.date(2020, 8, 15)
    freq = 4
    eom = 31
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.25")
    assert result == expected

def test_coupon_asof_on_start_date():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.0"))
    principal = Money(Decimal("1500"), Currency.USD)
    rate = Decimal("0.04")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 7, 1)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.0")
    assert result == expected

def test_coupon_asof_on_end_date():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("1.0"))
    principal = Money(Decimal("3000"), Currency.USD)
    rate = Decimal("0.06")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2020, 7, 1)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("1.0")
    assert result == expected

def test_coupon_frequency_one():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.75"))
    principal = Money(Decimal("2500"), Currency.USD)
    rate = Decimal("0.02")
    start = datetime.date(2020, 6, 1)
    asof = datetime.date(2020, 9, 1)
    end = datetime.date(2021, 6, 1)
    freq = 1
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.75")
    assert result == expected

def test_coupon_with_negative_rate():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("1000"), Currency.USD)
    rate = Decimal("-0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.5")
    assert result == expected

def test_coupon_zero_principal():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("0"), Currency.USD)
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 7, 1)
    end = datetime.date(2021, 1, 1)
    freq = 2
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.5")
    assert result == expected

def test_coupon_large_frequency():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"))
    principal = Money(Decimal("5000"), Currency.USD)
    rate = Decimal("0.08")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 2, 1)
    end = datetime.date(2020, 3, 1)
    freq = 12
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.1")
    assert result == expected


# LLM-generated content at query #8
#--------------------------

def test_dcfc_30_360_us_basic():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_leap_day():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_month_end_adjustment():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_multi_year():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    expected = Decimal('1.33333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_same_date():
    start = datetime.date(2023, 5, 15)
    asof = start
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    assert result == Decimal('0')

def test_dcfc_30_360_us_d1_last_day_of_month():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    d1 = 30
    d2 = 28
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_d2_last_day_of_month():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    d1 = 30
    d2 = 30
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_d1_31_d2_31():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    d1 = 30
    d2 = 30
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_d1_30_d2_31():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 3, 31)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    d1 = 30
    d2 = 30
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_d1_31_d2_30():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 30)
    end = asof
    result = dcfc_30_360_us(start, asof, end)
    d1 = 30
    d2 = 30
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected


# LLM-generated content at query #9
#--------------------------

def test_dcfc_30_360_us_basic_examples():
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    result1 = round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14)
    result2 = round(dcfc_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14)
    result3 = round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14)
    result4 = round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14)
    assert result1 == Decimal('0.16666666666667')
    assert result2 == Decimal('0.16944444444444')
    assert result3 == Decimal('1.08333333333333')
    assert result4 == Decimal('1.33333333333333')

def test_dcfc_30_360_us_same_day():
    start = datetime.date(2020, 1, 15)
    asof = datetime.date(2020, 1, 15)
    end = datetime.date(2020, 1, 15)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    assert result == Decimal('0')

def test_dcfc_30_360_us_last_day_of_month_start():
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 28)
    end = datetime.date(2020, 2, 28)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal((30 - 30) + 30 * (2 - 1) + 360 * (2020 - 2020)) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_last_day_of_month_asof():
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal((30 - 28) + 30 * (2 - 2) + 360 * (2020 - 2020)) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_both_last_day_of_month():
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal((30 - 30) + 30 * (2 - 1) + 360 * (2020 - 2020)) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_d1_31_d2_31():
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 31)
    end = datetime.date(2020, 2, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal((30 - 30) + 30 * (2 - 1) + 360 * (2020 - 2020)) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_d1_30_d2_31():
    start = datetime.date(2020, 1, 30)
    asof = datetime.date(2020, 2, 31)
    end = datetime.date(2020, 2, 31)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal((30 - 30) + 30 * (2 - 1) + 360 * (2020 - 2020)) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_d1_31_d2_30():
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 2, 30)
    end = datetime.date(2020, 2, 30)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal((30 - 30) + 30 * (2 - 1) + 360 * (2020 - 2020)) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_cross_year():
    start = datetime.date(2019, 12, 15)
    asof = datetime.date(2020, 1, 15)
    end = datetime.date(2020, 1, 15)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal((15 - 15) + 30 * (1 - 12) + 360 * (2020 - 2019)) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_negative_days():
    start = datetime.date(2020, 2, 15)
    asof = datetime.date(2020, 1, 15)
    end = datetime.date(2020, 1, 15)
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    assert result == Decimal('0')


# LLM-generated content at query #10
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

def test_last_payment_date_semiannual_before_mid_year():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2)
    expected = datetime.date(2015, 7, 1)
    assert result == expected

def test_last_payment_date_semiannual_early_year():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_annual_start_mid_year():
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

def test_last_payment_date_semiannual_december_end():
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    expected = datetime.date(2015, 12, 15)
    assert result == expected

def test_last_payment_date_with_eom_override():
    result = _last_payment_date(datetime.date(2014, 1, 15), datetime.date(2015, 12, 31), 1, eom=31)
    expected = datetime.date(2015, 1, 31)
    assert result == expected

def test_last_payment_date_monthly_frequency():
    result = _last_payment_date(datetime.date(2014, 1, 10), datetime.date(2015, 12, 31), 12)
    expected = datetime.date(2015, 12, 10)
    assert result == expected

def test_last_payment_date_biannual_frequency():
    result = _last_payment_date(datetime.date(2014, 3, 20), datetime.date(2015, 9, 20), 2)
    expected = datetime.date(2015, 3, 20)
    assert result == expected

def test_last_payment_date_asof_before_first_payment():
    result = _last_payment_date(datetime.date(2015, 6, 1), datetime.date(2015, 5, 31), 1)
    expected = datetime.date(2015, 6, 1)
    assert result == expected

def test_last_payment_date_frequency_zero_division_handling():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), Decimal('0.5'))
    expected = datetime.date(2015, 1, 1)
    assert result == expected


# LLM-generated content at query #11
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


# LLM-generated content at query #12
#--------------------------

def test_last_payment_date_annual_frequency():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_same_year_annual():
    result = _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_semi_annual_frequency():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_before_mid_year():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2)
    assert result == datetime.date(2015, 7, 1)

def test_last_payment_date_semi_annual_early_year():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_annual_start_mid_year():
    result = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1)
    assert result == datetime.date(2014, 6, 1)

def test_last_payment_date_quarterly_frequency():
    result = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4)
    assert result == datetime.date(2015, 7, 7)

def test_last_payment_date_annual_december_start():
    result = _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1)
    assert result == datetime.date(2014, 12, 9)

def test_last_payment_date_semi_annual_december_start():
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2)
    assert result == datetime.date(2015, 12, 15)

def test_last_payment_date_semi_annual_december_end_year():
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 12, 15)

def test_last_payment_date_with_eom_override():
    result = _last_payment_date(datetime.date(2014, 1, 31), datetime.date(2015, 12, 31), 1, eom=31)
    assert result == datetime.date(2015, 1, 31)

def test_last_payment_date_february_eom_adjustment():
    result = _last_payment_date(datetime.date(2014, 2, 30), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 2, 28)

def test_last_payment_date_monthly_frequency():
    result = _last_payment_date(datetime.date(2014, 1, 15), datetime.date(2015, 12, 31), 12)
    assert result == datetime.date(2015, 12, 15)

def test_last_payment_date_before_first_payment():
    result = _last_payment_date(datetime.date(2015, 6, 1), datetime.date(2015, 5, 31), 1)
    assert result == datetime.date(2015, 6, 1)

def test_last_payment_date_exact_payment_day():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 1, 1), 1)
    assert result == datetime.date(2015, 1, 1)


# LLM-generated content at query #13
#--------------------------

def test_dcfc_30_e_360_basic_example_1():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_basic_example_2():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_basic_example_3():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_basic_example_4():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal('1.33055555555556')
    assert round(result, 14) == expected

def test_dcfc_30_e_360_start_day_31_adjusted():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_e_360(start, asof, end)
    adjusted_start = datetime.date(2023, 1, 30)
    nod = (asof.day - adjusted_start.day) + 30 * (asof.month - adjusted_start.month) + 360 * (asof.year - adjusted_start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_asof_day_31_adjusted():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 3, 31)
    end = datetime.date(2023, 3, 31)
    result = dcfc_30_e_360(start, asof, end)
    adjusted_asof = datetime.date(2023, 3, 30)
    nod = (adjusted_asof.day - start.day) + 30 * (adjusted_asof.month - start.month) + 360 * (adjusted_asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_both_days_31_adjusted():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    end = datetime.date(2023, 3, 31)
    result = dcfc_30_e_360(start, asof, end)
    adjusted_start = datetime.date(2023, 1, 30)
    adjusted_asof = datetime.date(2023, 3, 30)
    nod = (adjusted_asof.day - adjusted_start.day) + 30 * (adjusted_asof.month - adjusted_start.month) + 360 * (adjusted_asof.year - adjusted_start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_same_date():
    start = datetime.date(2023, 5, 15)
    asof = datetime.date(2023, 5, 15)
    end = datetime.date(2023, 5, 15)
    result = dcfc_30_e_360(start, asof, end)
    expected = Decimal(0)
    assert result == expected

def test_dcfc_30_e_360_cross_year():
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 1, 15)
    end = datetime.date(2023, 1, 15)
    result = dcfc_30_e_360(start, asof, end)
    nod = (asof.day - start.day) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_e_360_leap_year_feb_29():
    start = datetime.date(2024, 2, 28)
    asof = datetime.date(2024, 2, 29)
    end = datetime.date(2024, 2, 29)
    result = dcfc_30_e_360(start, asof, end)
    nod = (asof.day - start.day) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected


# LLM-generated content at query #14
#--------------------------

def test_calculate_daily_fraction_with_asof_after_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.5"))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.5")
    assert result == expected

def test_calculate_daily_fraction_with_asof_equal_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.5"))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 1)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.5")
    assert result == expected

def test_calculate_daily_fraction_with_asof_minus_one_before_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.5"))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 1)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.5")
    assert result == expected

def test_calculate_daily_fraction_with_asof_minus_one_after_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.5"))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.5")
    assert result == expected

def test_calculate_daily_fraction_with_freq():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.5"))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 2)
    end = Date(2023, 1, 3)
    freq = Decimal("2")
    result = dcc.calculate_daily_fraction(start, asof, end, freq)
    expected = Decimal("0.5")
    assert result == expected

def test_calculate_daily_fraction_with_asof_at_end():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("1.0"))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 3)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("1.0")
    assert result == expected

def test_calculate_daily_fraction_with_asof_after_end():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.0"))
    start = Date(2023, 1, 1)
    asof = Date(2023, 1, 4)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.0")
    assert result == expected

def test_calculate_daily_fraction_with_asof_before_start():
    dcc = DCC("Test", set(), set(), lambda s, a, e, f: Decimal("0.0"))
    start = Date(2023, 1, 2)
    asof = Date(2023, 1, 1)
    end = Date(2023, 1, 3)
    result = dcc.calculate_daily_fraction(start, asof, end)
    expected = Decimal("0.0")
    assert result == expected


# LLM-generated content at query #15
#--------------------------

def test_last_payment_date_annual_frequency():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_annual_frequency_same_year():
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

def test_last_payment_date_semi_annual_frequency_before_mid_year():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 8, 31)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 7, 1)
    assert result == expected

def test_last_payment_date_semi_annual_frequency_early_year():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 4, 30)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_annual_frequency_mid_year_start():
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

def test_last_payment_date_annual_frequency_december_start():
    start = datetime.date(2014, 12, 9)
    asof = datetime.date(2015, 12, 4)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2014, 12, 9)
    assert result == expected

def test_last_payment_date_semi_annual_frequency_december_start():
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2016, 1, 6)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 12, 15)
    assert result == expected

def test_last_payment_date_semi_annual_frequency_december_start_end_year():
    start = datetime.date(2012, 12, 15)
    asof = datetime.date(2015, 12, 31)
    frequency = 2
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 12, 15)
    assert result == expected

def test_last_payment_date_with_eom_parameter():
    start = datetime.date(2014, 1, 31)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    eom = 31
    result = _last_payment_date(start, asof, frequency, eom)
    expected = datetime.date(2015, 1, 31)
    assert result == expected

def test_last_payment_date_monthly_frequency():
    start = datetime.date(2014, 1, 15)
    asof = datetime.date(2015, 12, 20)
    frequency = 12
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 12, 15)
    assert result == expected

def test_last_payment_date_quarterly_frequency_feb_start():
    start = datetime.date(2014, 2, 28)
    asof = datetime.date(2015, 11, 30)
    frequency = 4
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 8, 28)
    assert result == expected

def test_last_payment_date_before_start():
    start = datetime.date(2015, 6, 1)
    asof = datetime.date(2015, 5, 31)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = start
    assert result == expected

def test_last_payment_date_frequency_decimal():
    start = datetime.date(2014, 1, 1)
    asof = datetime.date(2015, 12, 31)
    frequency = Decimal(1)
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_last_payment_date_invalid_date_handling():
    start = datetime.date(2014, 2, 30)
    asof = datetime.date(2015, 12, 31)
    frequency = 1
    result = _last_payment_date(start, asof, frequency)
    expected = datetime.date(2015, 2, 28)
    assert result == expected


# LLM-generated content at query #16
#--------------------------

def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #17
#--------------------------

def test_is_registered_returns_false_for_unregistered_name():
    registry = DCCRegistryMachinery()
    result = registry._is_registered("Act/Act")
    assert result == False


# LLM-generated content at query #18
#--------------------------

def test_next_payment_date_no_eom():
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, None)
    expected = datetime.date(2015, 1, 1)
    assert result == expected

def test_next_payment_date_with_eom():
    result = _next_payment_date(datetime.date(2014, 1, 1), 1, 15)
    expected = datetime.date(2015, 1, 15)
    assert result == expected

def test_next_payment_date_frequency_half_year():
    result = _next_payment_date(datetime.date(2014, 1, 1), 2, None)
    expected = datetime.date(2014, 7, 1)
    assert result == expected

def test_next_payment_date_frequency_quarter():
    result = _next_payment_date(datetime.date(2014, 1, 1), 4, None)
    expected = datetime.date(2014, 4, 1)
    assert result == expected

def test_next_payment_date_eom_invalid_day():
    result = _next_payment_date(datetime.date(2014, 1, 31), 1, 30)
    expected = datetime.date(2015, 1, 31)
    assert result == expected

def test_next_payment_date_eom_valid_day():
    result = _next_payment_date(datetime.date(2014, 1, 15), 1, 31)
    expected = datetime.date(2015, 1, 31)
    assert result == expected

def test_next_payment_date_frequency_decimal():
    result = _next_payment_date(datetime.date(2014, 1, 1), Decimal('0.5'), None)
    expected = datetime.date(2026, 1, 1)
    assert result == expected

def test_next_payment_date_frequency_int_three():
    result = _next_payment_date(datetime.date(2014, 1, 1), 3, None)
    expected = datetime.date(2014, 5, 1)
    assert result == expected

def test_next_payment_date_cross_year_boundary():
    result = _next_payment_date(datetime.date(2014, 11, 1), 2, None)
    expected = datetime.date(2015, 5, 1)
    assert result == expected

def test_next_payment_date_leap_year():
    result = _next_payment_date(datetime.date(2020, 2, 29), 1, None)
    expected = datetime.date(2021, 2, 28)
    assert result == expected


# LLM-generated content at query #19
#--------------------------

def test_find_existing_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    result = registry.find("Act/Act")
    assert result == dcc

def test_find_existing_alt_name():
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
    dcc = DCC(name="ACT/ACT", altnames=["ACTUAL/ACTUAL"])
    registry.register(dcc)
    result = registry.find(" actual/actual ")
    assert result == dcc

def test_find_non_existing_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    result = registry.find("NonExistent")
    assert result is None

def test_find_empty_string():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    result = registry.find("")
    assert result is None

def test_find_case_sensitive_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    result = registry.find("act/act")
    assert result is None

def test_find_case_sensitive_alt_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=["Actual/Actual"])
    registry.register(dcc)
    result = registry.find("actual/actual")
    assert result is None

def test_find_with_whitespace_only():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="ACT/ACT", altnames=[])
    registry.register(dcc)
    result = registry.find("   ")
    assert result is None


# LLM-generated content at query #20
#--------------------------

def test_last_payment_date_annual_frequency():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_same_year_annual():
    result = _last_payment_date(datetime.date(2015, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_semiannual_frequency():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 7, 1)

def test_last_payment_date_semiannual_before_mid_year():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 8, 31), 2)
    assert result == datetime.date(2015, 7, 1)

def test_last_payment_date_semiannual_early_year():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 4, 30), 2)
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_annual_start_mid_year():
    result = _last_payment_date(datetime.date(2014, 6, 1), datetime.date(2015, 4, 30), 1)
    assert result == datetime.date(2014, 6, 1)

def test_last_payment_date_quarterly_frequency():
    result = _last_payment_date(datetime.date(2008, 7, 7), datetime.date(2015, 10, 6), 4)
    assert result == datetime.date(2015, 7, 7)

def test_last_payment_date_annual_december_start():
    result = _last_payment_date(datetime.date(2014, 12, 9), datetime.date(2015, 12, 4), 1)
    assert result == datetime.date(2014, 12, 9)

def test_last_payment_date_semiannual_december_start():
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2016, 1, 6), 2)
    assert result == datetime.date(2015, 12, 15)

def test_last_payment_date_semiannual_december_end_of_year():
    result = _last_payment_date(datetime.date(2012, 12, 15), datetime.date(2015, 12, 31), 2)
    assert result == datetime.date(2015, 12, 15)

def test_last_payment_date_with_eom_override():
    result = _last_payment_date(datetime.date(2014, 1, 15), datetime.date(2015, 12, 31), 1, eom=31)
    assert result == datetime.date(2015, 1, 31)

def test_last_payment_date_frequency_decimal():
    result = _last_payment_date(datetime.date(2014, 1, 1), datetime.date(2015, 12, 31), Decimal(1))
    assert result == datetime.date(2015, 1, 1)

def test_last_payment_date_edge_case_year_rollover():
    result = _last_payment_date(datetime.date(2014, 12, 31), datetime.date(2015, 1, 1), 12)
    assert result == datetime.date(2014, 12, 31)

def test_last_payment_date_monthly_frequency():
    result = _last_payment_date(datetime.date(2014, 1, 15), datetime.date(2015, 3, 20), 12)
    assert result == datetime.date(2015, 3, 15)

def test_last_payment_date_biannual_frequency():
    result = _last_payment_date(datetime.date(2014, 6, 30), datetime.date(2015, 11, 15), 2)
    assert result == datetime.date(2015, 6, 30)

def test_last_payment_date_start_date_after_asof():
    result = _last_payment_date(datetime.date(2016, 1, 1), datetime.date(2015, 12, 31), 1)
    assert result == datetime.date(2016, 1, 1)


# LLM-generated content at query #21
#--------------------------

def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


# LLM-generated content at query #22
#--------------------------

def test_is_registered_returns_false_for_unregistered_name():
    registry = DCCRegistryMachinery()
    result = registry._is_registered("Act/Act")
    assert result == False


# LLM-generated content at query #23
#--------------------------

def test_register_raises_type_error_when_altname_already_registered():
    from collections import namedtuple
    from decimal import Decimal
    from typing import Set, Optional
    Currency = namedtuple('Currency', ['code'])
    DCFC = namedtuple('DCFC', [])
    Date = namedtuple('Date', [])
    ZERO = Decimal('0')
    DCC = namedtuple('DCC', ['name', 'altnames', 'currencies', 'calculate_fraction_method'])
    registry = DCCRegistryMachinery()
    dummy_currency = Currency(code='USD')
    dummy_method = DCFC()
    dcc1 = DCC(name='First', altnames={'AltName'}, currencies={dummy_currency}, calculate_fraction_method=dummy_method)
    dcc2 = DCC(name='Second', altnames={'AltName'}, currencies={dummy_currency}, calculate_fraction_method=dummy_method)
    registry.register(dcc1)
    raised_exception = None
    try:
        registry.register(dcc2)
    except TypeError as e:
        raised_exception = e
    assert raised_exception is not None
    assert str(raised_exception) == "Day count convention 'Second' is already registered"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_dcfc_30_360_isda_example_1():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = datetime.date(2008, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

def test_dcfc_30_360_isda_example_2():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = datetime.date(2008, 2, 29)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

def test_dcfc_30_360_isda_example_3():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = datetime.date(2008, 11, 30)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_360_isda_example_4():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = datetime.date(2009, 5, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('1.33333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_360_isda_start_day_31():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = datetime.date(2023, 2, 28)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('0.07777777777778')
    assert round(result, 14) == expected

def test_dcfc_30_360_isda_start_day_30_asof_day_31():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 31)
    end = datetime.date(2023, 2, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('0.08611111111111')
    assert round(result, 14) == expected

def test_dcfc_30_360_isda_same_date():
    start = datetime.date(2023, 5, 15)
    asof = datetime.date(2023, 5, 15)
    end = datetime.date(2023, 5, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('0')
    assert result == expected

def test_dcfc_30_360_isda_cross_year():
    start = datetime.date(2022, 12, 15)
    asof = datetime.date(2023, 3, 15)
    end = datetime.date(2023, 3, 15)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('0.25')
    assert result == expected

def test_dcfc_30_360_isda_leap_year():
    start = datetime.date(2020, 2, 28)
    asof = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('0.00277777777778')
    assert round(result, 14) == expected

def test_dcfc_30_360_isda_month_end_adjustment():
    start = datetime.date(2023, 4, 30)
    asof = datetime.date(2023, 5, 31)
    end = datetime.date(2023, 5, 31)
    result = dcfc_30_360_isda(start=start, asof=asof, end=end)
    expected = Decimal('0.08333333333333')
    assert round(result, 14) == expected


# LLM-generated content at query #2
#--------------------------

def test_register_successfully_registers_new_dcc():
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

def test_register_raises_error_if_altname_already_registered_as_main():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="MainDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc1)
    dcc2 = DCC(name="TestDCC", altnames={"MainDCC"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'TestDCC' is already registered"

def test_register_raises_error_if_altname_already_registered_as_altname():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="DCC1", altnames={"ALT"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc1)
    dcc2 = DCC(name="DCC2", altnames={"ALT"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.3"))
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'DCC2' is already registered"

def test_register_adds_to_main_buffer():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TD"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._buffer_main["TestDCC"] == dcc

def test_register_adds_altnames_to_alt_buffer():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TD", "Test"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._buffer_altn["TD"] == dcc
    assert registry._buffer_altn["Test"] == dcc

def test_register_does_not_add_main_name_to_alt_buffer():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames={"TD"}, currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert "TestDCC" not in registry._buffer_altn

def test_register_with_empty_altnames():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="TestDCC", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    registry.register(dcc)
    assert registry._find_strict("TestDCC") == dcc
    assert len(registry._buffer_altn) == 0


# LLM-generated content at query #3
#--------------------------

def test_has_leap_day_with_leap_day_in_range():
    start = datetime.date(2020, 2, 28)
    end = datetime.date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_with_leap_day_at_start():
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 3, 1)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_with_leap_day_at_end():
    start = datetime.date(2020, 2, 28)
    end = datetime.date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_without_leap_day():
    start = datetime.date(2021, 2, 28)
    end = datetime.date(2021, 3, 1)
    result = _has_leap_day(start, end)
    assert result == False

def test_has_leap_day_across_multiple_years_with_leap_day():
    start = datetime.date(2019, 12, 31)
    end = datetime.date(2021, 1, 1)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_across_multiple_years_without_leap_day():
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2022, 12, 31)
    result = _has_leap_day(start, end)
    assert result == False

def test_has_leap_day_single_day_non_leap():
    start = datetime.date(2021, 2, 28)
    end = datetime.date(2021, 2, 28)
    result = _has_leap_day(start, end)
    assert result == False

def test_has_leap_day_single_day_leap():
    start = datetime.date(2020, 2, 29)
    end = datetime.date(2020, 2, 29)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_range_in_leap_year_but_not_including_feb29():
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 2, 28)
    result = _has_leap_day(start, end)
    assert result == False

def test_has_leap_day_range_starting_after_leap_day():
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 12, 31)
    result = _has_leap_day(start, end)
    assert result == False

def test_has_leap_day_range_ending_before_leap_day():
    start = datetime.date(2020, 1, 1)
    end = datetime.date(2020, 2, 28)
    result = _has_leap_day(start, end)
    assert result == False

def test_has_leap_day_range_spanning_leap_year_boundary():
    start = datetime.date(2019, 6, 1)
    end = datetime.date(2021, 6, 1)
    result = _has_leap_day(start, end)
    assert result == True

def test_has_leap_day_range_with_multiple_leap_days():
    start = datetime.date(2016, 1, 1)
    end = datetime.date(2024, 12, 31)
    result = _has_leap_day(start, end)
    assert result == True


# LLM-generated content at query #4
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

def test_dcfc_act_act_multiple_years():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('1.08243131970956')
    assert round(result, 14) == expected

def test_dcfc_act_act_another_multiple_years():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('1.32625945055768')
    assert round(result, 14) == expected

def test_dcfc_act_act_same_date():
    start = datetime.date(2020, 1, 1)
    asof = start
    end = start
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
    expected = Decimal('1') / Decimal(365) + Decimal('0') / Decimal(366)
    assert result == expected

def test_dcfc_act_act_full_leap_year():
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('366') / Decimal(366)
    assert result == expected

def test_dcfc_act_act_full_non_leap_year():
    start = datetime.date(2019, 1, 1)
    asof = datetime.date(2020, 1, 1)
    end = asof
    result = dcfc_act_act(start, asof, end)
    expected = Decimal('365') / Decimal(365)
    assert result == expected


# LLM-generated content at query #5
#--------------------------

def test_register_raises_type_error_when_main_name_already_registered():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Act/Act", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"))
    registry.register(dcc1)
    dcc2 = DCC(name="Act/Act", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.2"))
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'Act/Act' is already registered"


# LLM-generated content at query #6
#--------------------------

def test_dcfc_act_act_icma_basic():
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal('1'))
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

def test_dcfc_act_act_icma_same_date():
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 1)
    end = datetime.date(2017, 1, 1)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal('1'))
    assert result == Decimal('0')

def test_dcfc_act_act_icma_freq_parameter():
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal('2'))
    expected = Decimal('0.2622950820')
    assert round(result, 10) == expected

def test_dcfc_act_act_icma_freq_none():
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 9, 10)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=None)
    expected = Decimal('0.5245901639')
    assert round(result, 10) == expected

def test_dcfc_act_act_icma_asof_before_start():
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2019, 2, 1)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal('1'))
    assert result == Decimal('0')

def test_dcfc_act_act_icma_asof_after_end():
    start = datetime.date(2019, 3, 2)
    asof = datetime.date(2020, 4, 1)
    end = datetime.date(2020, 3, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal('1'))
    assert result == Decimal('0')

def test_dcfc_act_act_icma_one_day_period():
    start = datetime.date(2017, 1, 1)
    asof = datetime.date(2017, 1, 2)
    end = datetime.date(2017, 1, 2)
    result = dcfc_act_act_icma(start=start, asof=asof, end=end, freq=Decimal('1'))
    assert result == Decimal('1')


# LLM-generated content at query #7
#--------------------------

def test_coupon_regular_payment_schedule():
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
    start = datetime.date(2020, 2, 15)
    asof = datetime.date(2020, 8, 15)
    end = datetime.date(2021, 2, 15)
    freq = Decimal("1")
    eom = 31
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.25")
    assert result == expected

def test_coupon_asof_before_start():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.0"))
    principal = Money(Decimal("500"), Currency("GBP"))
    rate = Decimal("0.04")
    start = datetime.date(2021, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2022, 1, 1)
    freq = Decimal("2")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.0")
    assert result == expected

def test_coupon_asof_equals_start():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.0"))
    principal = Money(Decimal("1500"), Currency("JPY"))
    rate = Decimal("0.02")
    start = datetime.date(2020, 3, 1)
    asof = datetime.date(2020, 3, 1)
    end = datetime.date(2021, 3, 1)
    freq = Decimal("4")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.0")
    assert result == expected

def test_coupon_asof_equals_end():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("1.0"))
    principal = Money(Decimal("3000"), Currency("CAD"))
    rate = Decimal("0.06")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 1, 1)
    freq = Decimal("1")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("1.0")
    assert result == expected

def test_coupon_fraction_zero():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.0"))
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 1, 2)
    end = datetime.date(2021, 1, 1)
    freq = Decimal("2")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.0")
    assert result == expected

def test_coupon_fraction_full_period():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("1.0"))
    principal = Money(Decimal("2000"), Currency("EUR"))
    rate = Decimal("0.03")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 12, 31)
    end = datetime.date(2021, 1, 1)
    freq = Decimal("1")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("1.0")
    assert result == expected

def test_coupon_with_high_frequency():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.125"))
    principal = Money(Decimal("5000"), Currency("GBP"))
    rate = Decimal("0.04")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 4, 1)
    end = datetime.date(2021, 1, 1)
    freq = Decimal("12")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.125")
    assert result == expected

def test_coupon_with_eom_invalid_day():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.333"))
    principal = Money(Decimal("1000"), Currency("USD"))
    rate = Decimal("0.05")
    start = datetime.date(2020, 1, 31)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 31)
    freq = Decimal("2")
    eom = 31
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.333")
    assert result == expected

def test_coupon_negative_rate():
    dcc = DCC(name="test", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.5"))
    principal = Money(Decimal("1000"), Currency("EUR"))
    rate = Decimal("-0.02")
    start = datetime.date(2020, 1, 1)
    asof = datetime.date(2020, 6, 30)
    end = datetime.date(2021, 1, 1)
    freq = Decimal("2")
    eom = None
    result = dcc.coupon(principal, rate, start, asof, end, freq, eom)
    expected = principal * rate * Decimal("0.5")
    assert result == expected


# LLM-generated content at query #8
#--------------------------

def test_dcfc_act_365_a_basic_fraction():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    expected = Decimal('0.16986301369863')
    assert round(result, 14) == expected

def test_dcfc_act_365_a_leap_year_inclusion():
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

def test_dcfc_act_365_a_multi_year():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_act_365_a(start, asof, end)
    expected = Decimal('1.32513661202186')
    assert round(result, 14) == expected

def test_dcfc_act_365_a_same_date():
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
    expected = Decimal('2') / Decimal(365)
    assert result == expected

def test_dcfc_act_365_asof_before_start():
    start = datetime.date(2020, 1, 10)
    asof = datetime.date(2020, 1, 5)
    end = datetime.date(2020, 1, 15)
    result = dcfc_act_365_a(start, asof, end)
    assert result == Decimal('0')

def test_dcfc_act_365_a_asof_after_end():
    start = datetime.date(2020, 1, 10)
    asof = datetime.date(2020, 1, 20)
    end = datetime.date(2020, 1, 15)
    result = dcfc_act_365_a(start, asof, end)
    assert result == Decimal('0')


# LLM-generated content at query #9
#--------------------------

def test_dcfc_30_360_us_example1():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 28)
    end = asof
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal('0.16666666666667')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_example2():
    start = datetime.date(2007, 12, 28)
    asof = datetime.date(2008, 2, 29)
    end = asof
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal('0.16944444444444')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_example3():
    start = datetime.date(2007, 10, 31)
    asof = datetime.date(2008, 11, 30)
    end = asof
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal('1.08333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_example4():
    start = datetime.date(2008, 2, 1)
    asof = datetime.date(2009, 5, 31)
    end = asof
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    expected = Decimal('1.33333333333333')
    assert round(result, 14) == expected

def test_dcfc_30_360_us_same_date():
    start = datetime.date(2023, 1, 15)
    asof = start
    end = start
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    assert result == Decimal('0')

def test_dcfc_30_360_us_end_of_month_adjustment():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    d1 = 30
    d2 = 28
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_both_end_of_month():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 3, 31)
    end = asof
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    d1 = 30
    d2 = 30
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_d1_31_d2_31():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    d1 = 30
    d2 = 28
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_d1_30_d2_31():
    start = datetime.date(2023, 1, 30)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    d1 = 30
    d2 = 28
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected

def test_dcfc_30_360_us_d1_31_d2_30():
    start = datetime.date(2023, 1, 31)
    asof = datetime.date(2023, 2, 28)
    end = asof
    result = dcfc_30_360_us(start=start, asof=asof, end=end)
    d1 = 30
    d2 = 28
    nod = (d2 - d1) + 30 * (asof.month - start.month) + 360 * (asof.year - start.year)
    expected = Decimal(nod) / Decimal(360)
    assert result == expected


# LLM-generated content at query #10
#--------------------------

def test_find_existing_main_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    result = registry.find("Act/Act")
    assert result == dcc

def test_find_existing_alternative_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=["Actual/Actual"])
    registry.register(dcc)
    result = registry.find("Actual/Actual")
    assert result == dcc

def test_find_with_stripped_and_uppercased_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    result = registry.find("  act/act  ")
    assert result == dcc

def test_find_non_existing_name():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    result = registry.find("NonExistent")
    assert result is None

def test_find_case_insensitive_after_normalization():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="ACT/ACT", altnames=[])
    registry.register(dcc)
    result = registry.find("act/act")
    assert result == dcc

def test_find_empty_string():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    result = registry.find("")
    assert result is None

def test_find_whitespace_only():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=[])
    registry.register(dcc)
    result = registry.find("   ")
    assert result is None

def test_find_alternative_name_with_whitespace():
    registry = DCCRegistryMachinery()
    dcc = DCC(name="Act/Act", altnames=["  Actual/Actual  "])
    registry.register(dcc)
    result = registry.find("Actual/Actual")
    assert result == dcc


# LLM-generated content at query #11
#--------------------------

def test_register_raises_type_error_when_main_name_already_registered():
    registry = DCCRegistryMachinery()
    dcc1 = DCC(name="Act/Act", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.1"))
    registry.register(dcc1)
    dcc2 = DCC(name="Act/Act", altnames=set(), currencies=set(), calculate_fraction_method=lambda s, a, e, f: Decimal("0.2"))
    try:
        registry.register(dcc2)
        assert False
    except TypeError as e:
        assert str(e) == "Day count convention 'Act/Act' is already registered"


# LLM-generated content at query #12
#--------------------------

def test_constructor_initializes_empty_buffers():
    registry = DCCRegistryMachinery()
    assert registry._buffer_main == {}
    assert registry._buffer_altn == {}


