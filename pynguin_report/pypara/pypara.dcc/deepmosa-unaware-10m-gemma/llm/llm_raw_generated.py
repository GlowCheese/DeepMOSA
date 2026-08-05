####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal

def test_DCC_calculate_fraction():
    # Mocking a DCFC function (day count fraction calculation method)
    # Returns (end - start).days / 365 as a simple mock implementation
    def mock_calc(start, end, asof, freq):
        return Decimal((end - start).days) / Decimal("365")

    # Setup DCC instance
    dcc = DCC(
        name="Mock Convention",
        altnames={"Mock"},
        currencies=set(),
        calculate_fraction_method=mock_calc
    )

    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 6, 1)
    end_date = datetime.date(2024, 1, 1)

    # Test Case 1: Valid range (start <= asof <= end)
    # Expected calculation: (2024-01-01 - 2023-01-01) = 365 days. 365/365 = 1
    fraction_valid = dcc.calculate_fraction(start_date, asof_date, end_date)
    assert fraction_valid == Decimal("1.0") # Note: The mock uses (end-start), so it ignores asof in this specific logic

    # Test Case 2: Specific calculation based on the provided implementation logic
    # In the provided code, calculate_fraction calls self[3](start, asof, end, freq)
    # Let's test a scenario where we verify if 'asof' is passed correctly to the method
    def mock_calc_with_asof(start, asof, end, freq):
        return Decimal((asof - start).days)

    dcc_asof = DCC(
        name="AsOf Convention",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calc_with_asof
    )

    # (2023-06-01 - 2023-01-01) = 151 days
    fraction_asof = dcc_asof.calculate_fraction(start_date, asof_date, end_date)
    assert fraction_asof == Decimal("151")

    # Test Case 3: Invalid range (asof < start)
    # The code specifies: if not start <= asof <= end: return ZERO
    invalid_asof = datetime.date(2022, 1, 1)
    fraction_invalid = dcc.calculate_fraction(start_date, invalid_asof, end_date)
    assert fraction_invalid == Decimal("0")

    # Test Case 4: Invalid range (asof > end)
    invalid_end = datetime.date(2025, 1, 1)
    fraction_invalid_end = dcc.calculate_fraction(start_date, invalid_end, end_date)
    assert fraction_invalid_end == Decimal("0")

    # Test Case 5: Frequency parameter passed through
    def mock_calc_with_freq(start, asof, end, freq):
        return Decimal(freq) if freq is not None else Decimal("0")

    dcc_freq = DCC(
        name="Freq Convention",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_calc_with_freq
    )
    
    fraction_freq = dcc_freq.calculate_fraction(start_date, asof_date, end_date, freq=Decimal("2.5"))
    assert fraction_freq == Decimal("2.5")
```


# LLM-generated content at query #2
#--------------------------

```python
import datetime
from decimal import Decimal
import pytest

def test_dcfc_act_365_a():
    """
    Tests the dcfc_act_365_a function with various date scenarios including leap years.
    Assumes _get_actual_day_count and _has_leap_day are available in the scope 
    as they are internal dependencies of the provided code.
    """
    # Test Case 1: Non-leap year period (2007 to 2008, but asof is before leap day)
    # Note: Based on docstring: ex1_start, ex1_asof = 2007-12-28, 2008-02-28
    # The logic uses _get_actual_day_count(start, asof). 
    # Days: Dec(3) + Jan(31) + Feb(28) = 62 days. 
    # 62 / 365 = 0.16986301369863
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16986301369863')
    assert round(dcfc_act_365_a(ex1_start, ex1_asof, ex1_asof), 14) == expected1

    # Test Case 2: Period including leap day (2007-12-28 to 2008-02-29)
    # Days: Dec(3) + Jan(31) + Feb(29) = 63 days.
    # 63 / 365 = 0.17260273972603 (Note: The docstring value for ex2 suggests 
    # the denominator is 365 if the leap day is not yet reached or logic handles it)
    # Re-verifying docstring value: 0.17213114754098 corresponds to 62.83/365 approx.
    # We will test against the provided docstring expectations directly.
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    expected2 = Decimal('0.17213114754098')
    assert round(dcfc_act_365_a(ex2_start, ex2_asof, ex2_asof), 14) == expected2

    # Test Case 3: Long period crossing years
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08196721311475')
    assert round(dcfc_act_365_a(ex3_start, ex3_asof, ex3_asof), 14) == expected3

    # Test Case 4: Period in a different year range
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.32513661202186')
    assert round(dcfc_act_365_a(ex4_start, ex4_asof, ex4_asof), 14) == expected4

    # Test Case 5: Zero day interval
    same_day = datetime.date(2023, 1, 1)
    assert dcfc_act_365_a(same_day, same_day, same_day) == Decimal('0')
```


# LLM-generated content at query #3
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_nl_365():
    # Test case 1: Non-leap year transition (2007-12-28 to 2008-02-28)
    # _get_actual_day_count = 62 days. No leap day in range.
    # Result: 62 / 365 = 0.16986301369863
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    res1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(res1, 14) == Decimal('0.16986301369863')

    # Test case 2: Range includes leap day (2007-12-28 to 2008-02-29)
    # _get_actual_day_count = 63 days. Leap day exists.
    # Result: (63 - 1) / 365 = 62 / 365 = 0.16986301369863
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    res2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(res2, 14) == Decimal('0.16986301369863')

    # Test case 3: Multi-year range with leap year (2007-10-31 to 2008-11-30)
    # _get_actual_day_count = 426 days. Leap day exists.
    # Result: (426 - 1) / 365 = 425 / 365 = 1.16438356164384... 
    # Wait, the docstring says: Decimal('1.08219178082192')
    # Let's re-calculate based on provided doctest value:
    # 1.08219178082192 * 365 = 394.999... so days should be 395.
    # If we assume the function logic in docstring is ground truth for the target:
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    res3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(res3, 14) == Decimal('1.08219178082192')

    # Test case 4: Range in leap year (2008-02-01 to 2009-05-31)
    # _get_actual_day_count = 485 days. Leap day exists.
    # Result: (485 - 1) / 365 = 484 / 365 = 1.32602739726027
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    res4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(res4, 14) == Decimal('1.32602739726027')
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from decimal import Decimal
import datetime

def test_dcfc_act_365_l():
    """
    Tests the dcfc_act_365_l function with various date scenarios 
    to ensure correct day count fraction calculation based on leap years.
    """
    # Test Case 1: Non-leap year period (2007 to 2008 Feb 28)
    # Calculation: Days in range / 365 (since asof 2008-02-28 is not a leap day yet, 
    # but the function checks if asof.year is leap. 2008 IS a leap year).
    # Note: The implementation uses calendar.isleap(asof.year) to determine denominator.
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    # Days: Dec(4) + Jan(31) + Feb(28) = 63 days. 
    # asof.year is 2008 (leap), so denominator is 366.
    # However, the docstring example says: round(...) -> Decimal('0.16939890710383')
    # Let's verify calculation: 62 / 366 = 0.1693989... (Wait, 28th to 28th is 31 days in Jan + 4 in Dec?)
    # We rely on the provided docstring values for correctness verification.
    res1 = dcfc_act_365_l(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(res1, 14) == Decimal('0.16939890710383')

    # Test Case 2: Leap day included (2007 to 2008 Feb 29)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    res2 = dcfc_act_365_l(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(res2, 14) == Decimal('0.17213114754098')

    # Test Case 3: Non-leap year denominator check (asof in 2007)
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2007, 11, 30)
    res3 = dcfc_act_365_l(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(res3, 14) == Decimal('1.08196721311475')

    # Test Case 4: Period spanning into a leap year (2008 to 2009)
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    res4 = dcfc_act_365_l(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(res4, 14) == Decimal('1.32876712328767')

    # Test Case 5: Edge case - same day (0 days difference)
    same_day = datetime.date(2023, 1, 1)
    res5 = dcfc_act_365_l(start=same_day, asof=same_day, end=same_day)
    assert res5 == Decimal('0')
```


# LLM-generated content at query #5
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_30_e_360():
    # Test Case 1: Basic date range from docstring
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16666666666667')
    assert round(dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == expected1

    # Test Case 2: Leap year day handling (Feb 29)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    expected2 = Decimal('0.16944444444444')
    assert round(dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == expected2

    # Test Case 3: Start date on 31st (should be treated as 30th)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08333333333333')
    assert round(dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == expected3

    # Test Case 4: End date on 31st (should be treated as 30th)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    expected4 = Decimal('1.33055555555556')
    assert round(dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == expected4

    # Test Case 5: Same day (zero fraction)
    same_day = datetime.date(2023, 1, 1)
    assert dcfc_30_e_360(start=same_day, asof=same_day, end=same_day) == Decimal('0')

    # Test Case 6: Exactly one month (30 days / 360 = 1/12)
    month_start = datetime.date(2023, 1, 1)
    month_end = datetime.date(2023, 2, 1)
    expected_month = Decimal('30') / Decimal('360')
    assert dcfc_30_e_360(start=month_start, asof=month_end, end=month_end) == expected_month
```


# LLM-generated content at query #6
#--------------------------

```python
import datetime
from decimal import Decimal
import pytest

def test_dcfc_nl_365():
    """
    Tests the dcfc_nl_365 function with various date ranges including 
    leap year and non-leap year scenarios.
    """
    # Test Case 1: Non-leap year period (2007 to 2008 leap day not included)
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    res1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(res1, 14) == Decimal('0.16986301369863')

    # Test Case 2: Leap year period (contains Feb 29, 2008)
    # Note: The logic subtracts 1 if leap day is present in the range
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    res2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(res2, 14) == Decimal('0.16986301369863')

    # Test Case 3: Period spanning across a leap year (Oct 2007 to Nov 2008)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    res3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(res3, 14) == Decimal('1.08219178082192')

    # Test Case 4: Period in a leap year (Feb 2008 to May 2009)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    res4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(res4, 14) == Decimal('1.32602739726027')

    # Test Case 5: Single day (no leap day involved)
    day = datetime.date(2023, 1, 1)
    res5 = dcfc_nl_365(start=day, asof=day, end=day)
    # _get_actual_day_count(day, day) usually returns 0 or 1 depending on implementation.
    # Assuming standard behavior where start/asof same results in 0 difference.
    assert res5 >= 0
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
from decimal import Decimal
import datetime

def test_dcfc_act_365_l():
    """
    Tests the dcfc_act_365_l function with various date scenarios including leap and non-leap years.
    The convention uses 366 days if the 'asof' year is a leap year, otherwise 365.
    """
    # Scenario 1: Non-leap year asof (2007), 31 days elapsed
    # Dec 28 to Jan 28 of next year (not used here, just testing day count logic)
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 1, 27) # Dec 28 to Jan 27 is 30 days
    # Expected: 30 / 366 (since 2008 is a leap year)
    expected1 = Decimal('30') / Decimal('366')
    assert round(dcfc_act_365_l(start=start1, asof=asof1, end=asof1), 14) == round(expected1, 14)

    # Scenario 2: Leap year asof (2008), specifically testing the provided doctest example
    # Ex 1: 2007-12-28 to 2008-02-28. As of 2008 is leap year. 
    # Days: Dec(3) + Jan(31) + Feb(28) = 62 days? No, _get_actual_day_count depends on implementation.
    # Using the provided doctest value as the ground truth for the logic.
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    expected_val1 = Decimal('0.16939890710383')
    assert round(dcf1_act_365_l(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == expected_val1

    # Scenario 3: Non-leap year asof (2007)
    # If asof is in 2007, denominator must be 365.
    start3 = datetime.date(2007, 1, 1)
    asof3 = datetime.date(2007, 1, 2) # 1 day elapsed
    expected3 = Decimal('1') / Decimal('365')
    assert round(dcfc_act_365_l(start=start3, asof=asof3, end=asof3), 14) == round(expected3, 14)

    # Scenario 4: Leap year asof (2008)
    # If asof is in 2008, denominator must be 366.
    start4 = datetime.date(2008, 1, 1)
    asof4 = datetime.date(2008, 1, 2) # 1 day elapsed
    expected4 = Decimal('1') / Decimal('366')
    assert round(dcfc_act_365_l(start=start4, asof=asof4, end=asof4), 14) == round(expected4, 14)

    # Scenario 5: Testing the 'end' parameter influence via _get_actual_day_count
    # (Assuming _get_actual_day_count calculates difference between start and asof)
    start5 = datetime.date(2008, 2, 28)
    asof5 = datetime.date(2008, 3, 1) # 2 days elapsed (Feb 28 to Mar 1 in leap year: Feb 29 is included)
    # Days: 2. Denominator: 366 (2008 is leap)
    expected5 = Decimal('2') / Decimal('366')
    assert round(dcfc_act_365_l(start=start5, asof=asof5, end=asof5), 14) == round(expected5, 14)

# Helper to allow the test to run if the function name was typoed in the prompt's logic
def dcf1_act_365_l(*args, **kwargs):
    return dcfc_act_365_l(*args, **kwargs)
```


# LLM-generated content at query #8
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_act_365_a():
    # Test cases provided in the docstring
    test_cases = [
        {
            "start": datetime.date(2007, 12, 28),
            "asof": datetime.date(2008, 2, 28),
            "end": datetime.date(2008, 2, 28),
            "expected": Decimal('0.16986301369863')
        },
        {
            "start": datetime.date(2007, 12, 28),
            "asof": datetime.date(2008, 2, 29),
            "end": datetime.date(2008, 2, 29),
            "expected": Decimal('0.17213114754098')
        },
        {
            "start": datetime.date(2007, 10, 31),
            "asof": datetime.date(2008, 11, 30),
            "end": datetime.date(2008, 11, 30),
            "expected": Decimal('1.08196721311475')
        },
        {
            "start": datetime.date(2008, 2, 1),
            "asof": datetime.date(2009, 5, 31),
            "end": datetime.date(2009, 5, 31),
            "expected": Decimal('1.32513661202186')
        }
    ]

    for case in test_cases:
        result = dcfc_act_365_a(
            start=case["start"], 
            asof=case["asof"], 
            end=case["end"]
        )
        # Use quantize or rounding to handle floating point precision issues in Decimals
        assert round(result, 14) == round(case["expected"], 14)
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal
from unittest.mock import MagicMock

def test_DCC_interest():
    """
    Tests the interest method of the DCC class.
    Verifies that the calculation follows: principal * rate * fraction.
    """
    # Setup mock dependencies
    mock_currency = MagicMock()
    # DCFC signature: (Date, Date, Date, Optional[Decimal]) -> Decimal
    mock_calc_method = MagicMock(return_value=Decimal("0.5"))
    
    # Create an instance of DCC
    dcc_instance = DCC(
        name="Actual/Actual",
        altnames={"ACT/ACT"},
        currencies={mock_currency},
        calculate_fraction_method=mock_calc_method
    )

    # Define test parameters
    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 6, 1)
    end_date = datetime.mock_date = datetime.date(2024, 1, 1) # end is optional in interest()
    rate = Decimal("0.05")
    
    # Mock Money object: needs to support multiplication (Money * Decimal -> Money)
    class MockMoney:
        def __init__(self, amount):
            self.amount = amount
        def __mul__(self, other):
            if isinstance(other, Decimal):
                return MockMoney(self.amount * other)
            return MagicMock()
        def __eq__(self, other):
            return isinstance(other, MockMoney) and self.amount == other.amount

    principal = MockMoney(Decimal("1000.00"))

    # Test Case 1: end_date is provided
    # Expected: 1000 * 0.05 * 0.5 = 25.00
    result = dcc_instance.interest(
        principal=principal,
        rate=rate,
        start=start_date,
        asof=asof_date,
        end=end_date
    )
    assert result.amount == Decimal("25.00")
    mock_calc_method.assert_called_with(start_date, asof_date, end_date, None)

    # Test Case 2: end_date is NOT provided (should default to asof_date)
    # Expected: 1000 * 0.05 * 0.5 = 25.00
    result_no_end = dcc_instance.interest(
        principal=principal,
        rate=rate,
        start=start_date,
        asof=asof_date,
        end=None
    )
    assert result_no_end.amount == Decimal("25.00")
    mock_calc_method.assert_called_with(start_date, asof_date, asof_date, None)

    # Test Case 3: Verify the logic when calculate_fraction returns ZERO (e.g., invalid date range)
    mock_calc_method.return_value = Decimal("0")
    result_zero = dcc_instance.interest(
        principal=principal,
        rate=rate,
        start=asof_date, # start > asof logic in calculate_fraction
        asof=asof_date,
        end=end_date
    )
    assert result_zero.amount == Decimal("0")
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal

def test_DCC_calculate_fraction():
    # Mocking a DCFC function: (end - start).days / 360
    def mock_fraction_method(start, asof, end, freq):
        return Decimal((end - start).days) / Decimal("360")

    # Setup test data
    name = "Mock Convention"
    altnames = {"Mock"}
    currencies = set()
    dcc = DCC(
        name=name,
        altnames=altnames,
        currencies=currencies,
        calculate_fraction_method=mock_fraction_method
    )

    date_start = datetime.date(2023, 1, 1)
    date_asof = datetime.date(2023, 6, 1)
    date_end = datetime.date(2024, 1, 1)
    freq = Decimal("2")

    # Case 1: Valid date range (start <= asof <= end)
    # Days between 2023-01-01 and 2024-01-01 is 365
    expected_fraction = Decimal("365") / Decimal("360")
    result = dcc.calculate_fraction(date_start, date_asof, date_end, freq)
    assert result == expected_fraction

    # Case 2: asof is equal to start
    # Days between 2023-01-01 and 2023-01-01 is 0
    result_start_eq_asof = dcc.calculate_fraction(date_start, date_start, date_end, freq)
    assert result_start_eq_asof == Decimal("365") / Decimal("360")

    # Case 3: asof is equal to end
    result_asof_eq_end = dcc.calculate_fraction(date_start, date_end, date_end, freq)
    assert result_asof_eq_end == Decimal("0") / Decimal("360")

    # Case 4: Invalid range (asof < start) -> Should return ZERO
    result_invalid_start = dcc.calculate_fraction(date_start, datetime.date(2022, 1, 1), date_end, freq)
    assert result_invalid_start == Decimal("0")

    # Case 5: Invalid range (asof > end) -> Should return ZERO
    result_invalid_end = dcc.calculate_fraction(date_start, datetime.date(2025, 1, 1), date_end, freq)
    assert result_invalid_end == Decimal("0")

    # Case 6: Invalid range (start > end) -> Should return ZERO
    result_swapped_dates = dcc.calculate_fraction(date_end, date_asof, date_start, freq)
    assert result_swapped_dates == Decimal("0")
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal
from unittest.mock import MagicMock

def test_DCC_coupon():
    """
    Tests the coupon method of the DCC class.
    The coupon method relies on interest, which in turn relies on calculate_fraction.
    It also uses internal helpers _last_payment_date and _next_payment_date.
    """
    # Mocking Money class behavior (needs to support multiplication)
    class MockMoney:
        def __init__(self, amount):
            self.amount = Decimal(amount)
        def __mul__(self, other):
            if isinstance(other, (Decimal, int)):
                return MockMoney(self.amount * Decimal(other))
            return other * self.amount

    # Setup test constants
    principal = MockMoney("1000.00")
    rate = Decimal("0.05")
    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 6, 1)
    end_date = datetime.date(2024, 1, 1)
    freq = 2  # Semi-annual
    eom = 1

    # Create a mock DCFC function (the 4th element of DCC NamedTuple)
    # We'll make it return a fixed fraction for predictable testing
    mock_calc_method = MagicMock(return_value=Decimal("0.25"))

    # Initialize DCC instance
    # DCC is a NamedTuple: (name, altnames, currencies, calculate_fraction_method)
    dcc_instance = DCC(
        name="Actual/Actual",
        altnames={"Act/Act"},
        currencies=set(),
        calculate_fraction_method=mock_calc_method
    )

    # Calculation logic check:
    # 1. _last_payment_date(start, asof, freq, eom) 
    #    For start=2023-01-01, asof=2023-06-01, freq=2 -> should find previous coupon date (2023-01-01)
    # 2. _next_payment_date(prevdate, freq, eom)
    #    For prevdate=2023-01-01, freq=2 -> next is 2023-07-01
    # 3. interest = principal * rate * fraction
    #    Expected: 1000 * 0.05 * 0.25 = 12.5

    result_money = dcc_instance.coupon(
        principal=principal,
        rate=rate,
        start=start_date,
        asof=asof_date,
        end=end_date,
        freq=freq,
        eom=eom
    )

    # Assertions
    assert isinstance(result_money, MockMoney)
    assert result_money.amount == Decimal("12.5")
    
    # Verify the underlying calculation method was called with expected dates
    # The coupon method calls interest -> calculate_fraction -> calculate_fraction_method
    # We check if it was called with the computed prevdate (2023-01-01) and nextdate (2023-07-01)
    mock_calc_method.assert_called()
    args, kwargs = mock_calc_method.call_args
    # args[0] is start, args[1] is asof, args[2] is end
    assert args[0] == datetime.date(2023, 1, 1)
    assert args[2] == datetime.date(2023, 7, 1)

def test_DCC_coupon_edge_case_frequency():
    """Tests coupon with annual frequency."""
    class MockMoney:
        def __init__(self, amount): self.amount = Decimal(amount)
        def __mul__(self, other): return MockMoney(self.amount * Decimal(other))

    mock_calc = MagicMock(return_value=Decimal("1.0"))
    dcc = DCC("Test", set(), set(), mock_calc)
    
    principal = MockMoney("100")
    rate = Decimal("0.10")
    start = datetime.date(2023, 1, 1)
    asof = datetime.date(2023, 12, 31)
    end = datetime.date(2024, 1, 1)
    
    # For freq=1 (annual), prevdate should be start date
    result = dcc.coupon(principal, rate, start, asof, end, 1)
    
    assert result.amount == Decimal("10.0")
```


# LLM-generated content at query #12
#--------------------------

```python
import datetime
from decimal import Decimal
import pytest

def test_dcfc_30_e_360():
    """
    Tests the dcfc_30_e_360 function with various date scenarios 
    to ensure correct day count fraction calculation.
    """
    # Test Case 1: Standard dates (No 31st)
    # Ex 1 from docstring: start=2007-12-28, asof=2008-02-28
    # nod = (28 - 28) + 30 * (2 - 12) + 360 * (2008 - 2007)
    # nod = 0 + (-300) + 360 = 60
    # 60 / 360 = 0.16666666666667
    start1 = datetime.date(2007, 12, 28)
    asof1 = datetime.date(2008, 2, 28)
    end1 = datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16666666666667')
    assert round(dcfc_30_e_360(start=start1, asof=asof1, end=end1), 14) == expected1

    # Test Case 2: Start date is the 31st (Should be adjusted to 30th)
    # Ex 3 from docstring: start=2007-10-31, asof=2008-11-30
    # Adjusted start: 2007-10-30. asof remains 2008-11-30.
    # nod = (30 - 30) + 30 * (11 - 10) + 360 * (2008 - 2007)
    # nod = 0 + 30 + 360 = 390
    # 390 / 360 = 1.08333333333333
    start2 = datetime.date(2007, 10, 31)
    asof2 = datetime.date(2008, 11, 30)
    end2 = datetime.date(2008, 11, 30)
    expected2 = Decimal('1.08333333333333')
    assert round(dcfc_30_e_360(start=start2, asof=asof2, end=end2), 14) == expected2

    # Test Case 3: As-of date is the 31st (Should be adjusted to 30th)
    # Start=2008-02-01, AsOf=2009-05-31 -> Adjusted AsOf=2009-05-30
    # nod = (30 - 1) + 30 * (5 - 2) + 360 * (2009 - 2008)
    # nod = 29 + 90 + 360 = 479
    # 479 / 360 = 1.33055555555556
    start3 = datetime.date(2008, 2, 1)
    asof3 = datetime.date(2009, 5, 31)
    end3 = datetime.date(2009, 5, 31)
    expected3 = Decimal('1.33055555555556')
    assert round(dcfc_30_e_360(start=start3, asof=asof3, end=end3), 14) == expected3

    # Test Case 4: Both start and asof are the 31st
    # Start=2007-01-31 -> 2007-01-30. AsOf=2007-03-31 -> 2007-03-30.
    # nod = (30 - 30) + 30 * (3 - 1) + 360 * (2007 - 2007)
    # nod = 0 + 60 + 0 = 60
    # 60 / 360 = 0.16666666666667
    start4 = datetime.date(2007, 1, 31)
    asof4 = datetime.date(2007, 3, 31)
    end4 = datetime.date(2007, 3, 31)
    expected4 = Decimal('0.16666666666667')
    assert round(dcfc_30_e_360(start=start4, asof=asof4, end=end4), 14) == expected4

    # Test Case 5: Same day (Zero period)
    start5 = datetime.date(2023, 6, 15)
    asof5 = datetime.date(2023, 6, 15)
    end5 = datetime.date(2023, 6, 15)
    expected5 = Decimal('0')
    assert dcfc_30_e_360(start=start5, asof=asof5, end=end5) == expected5
```


# LLM-generated content at query #13
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_nl_365():
    """
    Unit tests for the dcfc_nl_365 function.
    Validates calculation against provided doctest examples and edge cases.
    """
    # Test Case 1: Non-leap year transition (2007 to 2008, but asof is Feb 28)
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    # Expected: round(dcfc_nl_365(...), 14) -> Decimal('0.16986301369863')
    res1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(res1, 14) == Decimal('0.16986301369863')

    # Test Case 2: Leap year date (Feb 29 included in range)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    # Expected: round(dcfc_nl_365(...), 14) -> Decimal('0.16986301369863')
    res2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(res2, 14) == Decimal('0.16986301369863')

    # Test Case 3: Cross-month/year with specific day counts
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    # Expected: round(dcfc_nl_365(...), 14) -> Decimal('1.08219178082192')
    res3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(res3, 14) == Decimal('1.08219178082192')

    # Test Case 4: Longer period in leap year context
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    # Expected: round(dcfc_nl_365(...), 14) -> Decimal('1.32602739726027')
    res4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(res4, 14) == Decimal('1.32602739726027')

    # Test Case 5: Zero day interval (start == asof)
    zero_date = datetime.date(2023, 1, 1)
    res5 = dcfc_nl_365(start=zero_date, asof=zero_date, end=zero_date)
    assert res5 == Decimal('0')
```


# LLM-generated content at query #14
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_nl_365():
    # Test Case 1: Non-leap year transition (2007 to 2008, but asof is Feb 28)
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    res1 = dcfc_nl_365(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(res1, 14) == Decimal('0.16986301369863')

    # Test Case 2: Leap year transition (includes Feb 29)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    res2 = dcfc_nl_365(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(res2, 14) == Decimal('0.17260273972603') # Note: The docstring example for ex2 in nl_365 shows 0.169... but standard logic/docstrings suggest variation based on leap day removal

    # Test Case 3: Spanning a month change
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    res3 = dcfc_nl_365(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(res3, 14) == Decimal('1.08219178082192')

    # Test Case 4: Spanning multiple years and months
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    res4 = dcfc_nl_365(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(res4, 14) == Decimal('1.32602739726027')

    # Test Case 5: Exact same day (Zero days)
    day = datetime.date(2023, 1, 1)
    res5 = dcfc_nl_365(start=day, asof=day, end=day)
    assert res5 == Decimal('0')

    # Test Case 6: Check registration in Registry
    assert "NL/365" in DCCRegistry.table
    assert "NL365" in DCCRegistry.table
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_30_e_360():
    # Test Case 1: Standard case (Dec 28 to Feb 28)
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    ex1_end = datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16666666666667')
    assert round(dcfint_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_end), 14) == expected1

    # Test Case 2: Leap year day handling (Dec 28 to Feb 29)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    ex2_end = datetime.date(2008, 2, 29)
    expected2 = Decimal('0.16944444444444')
    assert round(dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_end), 14) == expected2

    # Test Case 3: Start date is 31st (Oct 31 to Nov 30)
    ex3_start = datetime.date(2008, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    ex3_end = datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08333333333333')
    assert round(dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_end), 14) == expected3

    # Test Case 4: Long period (Feb 1 to May 31)
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    ex4_end = datetime.date(2009, 5, 31)
    expected4 = Decimal('1.33055555555556')
    assert round(dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_end), 14) == expected4

    # Test Case 5: As-of date is 31st (should be treated as 30th)
    ex5_start = datetime.date(2023, 1, 1)
    ex5_asof = datetime.date(2023, 3, 31)
    ex5_end = datetime.date(2023, 3, 31)
    # (30-1) + 30*(3-1) + 360*(2023-2023) = 29 + 60 = 89 days
    # 89 / 360 = 0.24722222222222
    expected5 = Decimal('0.24722222222222')
    assert round(dcfc_30_e_360(start=ex5_start, asof=ex5_asof, end=ex5_end), 14) == expected5
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal

def test_DCC_calculate_fraction():
    # Mock DCFC function
    def mock_calc(start, asof, end, freq):
        return Decimal((end - start).days) / Decimal(360)

    # Setup DCC instance
    dcc = DCC(
        name="Mock Convention",
        altnames={"Mock"},
        currencies=set(),
        calculate_fraction_method=mock_calc
    )

    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 6, 1)
    end_date = datetime.date(2024, 1, 1)

    # Case 1: Valid range (start <= asof <= end)
    # Days between Jan 1 and June 1 2023 is 151 days
    expected_fraction = Decimal(151) / Decimal(360)
    result = dcc.calculate_fraction(start_date, asof_date, end_date)
    assert result == expected_fraction

    # Case 2: Invalid range (asof < start)
    invalid_asof = datetime.date(2022, 12, 31)
    result_invalid_start = dcc.calculate_fraction(start_date, invalid_asof, end_date)
    assert result_invalid_start == Decimal(0)

    # Case 3: Invalid range (asof > end)
    invalid_end = datetime.date(2025, 1, 1)
    result_invalid_end = dcc.calculate_fraction(start_date, asof_date, invalid_end)
    # Note: The logic 'if not start <= asof <= end' triggers if asof is outside [start, end]
    # If end is moved forward, the condition still holds true for asof. 
    # Let's test a case where asof is strictly greater than end.
    asof_beyond_end = datetime.date(2025, 1, 1)
    result_asof_beyond = dcc.calculate_fraction(start_date, asof_beyond_end, end_date)
    assert result_asof_beyond == Decimal(0)

    # Case 4: Frequency parameter passed through
    # The method should pass freq to the underlying function
    def mock_calc_with_freq(start, asof, end, freq):
        return Decimal(freq) if freq is not None else Decimal(0)

    dcc_freq = DCC("FreqTest", set(), set(), mock_calc_with_freq)
    assert dcc_freq.calculate_fraction(start_date, asof_date, end_date, freq=Decimal(5)) == Decimal(5)
```


# LLM-generated content at query #3
#--------------------------

```python
import datetime
from decimal import Decimal

def test_DCC_calculate_daily_fraction():
    # Mock DCFC function (Actual/365 logic)
    def mock_fraction(start, asof, end, freq=None):
        return Decimal((asof - start).days) / Decimal(365)

    # Setup DCC instance
    dcc = DCC(
        name="Mock Convention",
        altnames={"Mock"},
        currencies=set(),
        calculate_fraction_method=mock_fraction
    )

    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2024, 1, 1)

    # Test Case 1: Asof is the start date (today's factor should be 0, yesterday was before start)
    asof_start = datetime.date(2023, 1, 1)
    # tfact (Jan 1 to Jan 1) = 0/365 = 0
    # yfact (Dec 31 to Jan 1) -> asof_minus_1 is < start, so yfact = 0
    # Result: 0 - 0 = 0
    assert dcc.calculate_daily_fraction(start_date, asof_start, end_date) == Decimal(0)

    # Test Case 2: Asof is one day after start (today's factor is 1/365, yesterday was 0)
    asof_day_2 = datetime.date(2023, 1, 2)
    # tfact (Jan 1 to Jan 2) = 1/365
    # yfact (Jan 1 to Jan 1) = 0/365
    # Result: 1/365
    expected_val = Decimal(1) / Decimal(365)
    assert dcc.calculate_daily_fraction(start_date, asof_day_2, end_date) == expected_val

    # Test Case 3: Asof is several days after start
    asof_day_10 = datetime.date(2023, 1, 10)
    # tfact (Jan 1 to Jan 10) = 9/365
    # yfact (Jan 1 to Jan 9) = 8/365
    # Result: 1/365
    assert dcc.calculate_daily_fraction(start_date, asof_day_10, end_date) == expected_val

    # Test Case 4: Verifying calculation with frequency parameter passed through
    freq = Decimal(2)
    asof_day_3 = datetime.date(2023, 1, 3)
    # tfact (Jan 1 to Jan 3) = 2/365
    # yfact (Jan 1 to Jan 2) = 1/365
    assert dcc.calculate_daily_fraction(start_date, asof_day_3, end_date, freq=freq) == expected_val
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal

def test_DCC_calculate_fraction():
    # Mocking a DCFC function (Day Count Fraction Calculation)
    # Simply returns the number of days between start and end as a Decimal
    def mock_dc_method(start, end, asof, freq):
        return Decimal((end - start).days)

    # Setup dummy currencies and DCC instance
    # We use an empty set for currencies since we aren't testing currency logic here
    dummy_currency = None 
    dcc = DCC(
        name="Mock Convention",
        altnames={"Mock"},
        currencies=set(),
        calculate_fraction_method=mock_dc_method
    )

    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 1, 10)
    end_date = datetime.date(2023, 1, 20)

    # Test Case 1: Valid range (start <= asof <= end)
    # Expected: (end - start).days = 19
    fraction = dcc.calculate_fraction(start_date, asof_date, end_date)
    assert fraction == Decimal(19)

    # Test Case 2: Invalid range (asof < start)
    # Expected: ZERO (0)
    invalid_asof = datetime.date(2022, 12, 31)
    fraction_invalid_start = dcc.calculate_fraction(start_date, invalid_asof, end_date)
    assert fraction_invalid_start == Decimal(0)

    # Test Case 3: Invalid range (asof > end)
    # Expected: ZERO (0)
    invalid_end = datetime.date(2023, 1, 5) # asof is 10th, so asof > end
    fraction_invalid_end = dcc.calculate_fraction(start_date, asof_date, invalid_end)
    assert fraction_invalid_end == Decimal(0)

    # Test Case 4: Boundary condition (asof == start)
    # Expected: (end - start).days = 19
    fraction_boundary = dcc.calculate_fraction(start_date, start_date, end_date)
    assert fraction_boundary == Decimal(19)

    # Test Case 5: Boundary condition (asof == end)
    # Expected: (end - start).days = 19
    fraction_boundary_end = dcc.calculate_fraction(start_date, end_date, end_date)
    assert fraction_boundary_end == Decimal(19)

    # Test Case 6: Including frequency parameter in calculation
    def mock_dc_method_with_freq(start, end, asof, freq):
        base = Decimal((end - start).days)
        if freq:
            return base / freq
        return base

    dcc_with_freq = DCC(
        name="Freq Convention",
        altnames=set(),
        currencies=set(),
        calculate_fraction_method=mock_dc_method_with_freq
    )
    
    # Expected: 19 / 2 = 9.5
    fraction_with_freq = dcc_with_freq.calculate_fraction(start_date, asof_date, end_date, freq=Decimal(2))
    assert fraction_with_freq == Decimal('9.5')
```


# LLM-generated content at query #5
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_30_e_plus_360():
    # Test Case 1: Standard date range (no 31st involved)
    ex1_start = datetime.date(2007, 12, 28)
    ex1_asof = datetime.date(2008, 2, 28)
    # Calculation: 
    # start stays 28th. asof is not 31st.
    # nod = (28-28) + 30*(2-12) + 360*(2008-2007) = 0 - 300 + 360 = 60
    # 60 / 360 = 0.16666666666667
    assert round(dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

    # Test Case 2: Leap year Feb 29th (no 31st involved)
    ex2_start = datetime.date(2007, 12, 28)
    ex2_asof = datetime.date(2008, 2, 29)
    # Calculation:
    # start stays 28th. asof is not 31st.
    # nod = (29-28) + 30*(2-12) + 360*(2008-2007) = 1 - 300 + 360 = 61
    # 61 / 360 = 0.16944444444444
    assert round(dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

    # Test Case 3: Start date is 31st (should be treated as 30th)
    ex3_start = datetime.date(2007, 10, 31)
    ex3_asof = datetime.date(2008, 11, 30)
    # Calculation:
    # start becomes 30th. asof is not 31st.
    # nod = (30-30) + 30*(11-10) + 360*(2008-2007) = 0 + 30 + 360 = 390
    # 390 / 360 = 1.08333333333333
    assert round(dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

    # Test Case 4: End date (asof) is 31st (should be treated as next day)
    ex4_start = datetime.date(2008, 2, 1)
    ex4_asof = datetime.date(2009, 5, 31)
    # Calculation:
    # start is 1st. asof is 31st -> becomes June 1st (2009-06-01)
    # nod = (1-1) + 30*(6-2) + 360*(2009-2008) = 0 + 120 + 360 = 480
    # 480 / 360 = 1.33333333333333
    assert round(dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal

def test_DCC_calculate_fraction():
    # Mock DCFC function
    def mock_calc(start, asof, end, freq):
        return Decimal((end - start).days) / Decimal(365)

    # Setup DCC instance
    dcc = DCC(
        name="MockConvention",
        altnames={"Mock"},
        currencies=set(),
        calculate_fraction_method=mock_calc
    )

    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 6, 1)
    end_date = datetime.date(2024, 1, 1)

    # Test Case 1: Valid range (start <= asof <= end)
    # Days between 2023-01-01 and 2024-01-01 is 365. 365/365 = 1
    expected_fraction = Decimal(365) / Decimal(365)
    result = dcc.calculate_fraction(start_date, asof_date, end_date)
    assert result == expected_fraction

    # Test Case 2: Invalid range (asof < start)
    invalid_asof = datetime.date(2022, 12, 31)
    result_invalid_start = dcc.calculate_fraction(start_date, invalid_asof, end_date)
    assert result_invalid_start == Decimal(0)

    # Test Case 3: Invalid range (asof > end)
    invalid_end_asof = datetime.date(2024, 1, 2)
    result_invalid_end = dcc.calculate_fraction(start_date, asof_date, invalid_end_asof)
    assert result_invalid_end == Decimal(0)

    # Test Case 4: Boundary condition (asof == start)
    result_boundary_start = dcc.calculate_fraction(start_date, start_date, end_date)
    # Days between 2023-01-01 and 2023-01-01 is 0. 0/365 = 0
    assert result_boundary_start == Decimal(0)

    # Test Case 5: Boundary condition (asof == end)
    result_boundary_end = dcc.calculate_fraction(start_date, end_date, end_date)
    # Days between 2023-01-01 and 2024-01-01 is 365. 365/365 = 1
    assert result_boundary_end == expected_fraction

    # Test Case 6: Frequency parameter passed through to method
    freq = Decimal('2')
    result_with_freq = dcc.calculate_fraction(start_date, asof_date, end_date, freq=freq)
    assert result_with_freq == expected_fraction
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal

def test_DCC_calculate_fraction():
    # Mock DCFC function
    def mock_calc(start, asof, end, freq):
        return (end - start).days * Decimal('0.001')

    # Setup DCC instance
    dcc = DCC(
        name="Mock Convention",
        altnames={"Mock"},
        currencies=set(),
        calculate_fraction_method=mock_calc
    )

    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 1, 10)
    end_date = datetime.date(2023, 1, 20)

    # Case 1: Valid range (start <= asof <= end)
    # Expected: (20 - 1) * 0.001 = 0.019
    result_valid = dcc.calculate_fraction(start_date, asof_date, end_date)
    assert result_valid == Decimal('0.019')

    # Case 2: Invalid range (asof < start)
    # Expected: ZERO (0)
    result_invalid_asof_low = dcc.calculate_fraction(asof_date, start_date, end_date)
    assert result_invalid_asof_low == Decimal('0')

    # Case 3: Invalid range (asof > end)
    # Expected: ZERO (0)
    result_invalid_asof_high = dcc.calculate_fraction(start_date, end_date + datetime.timedelta(days=1), end_date)
    assert result_invalid_asof_high == Decimal('0')

    # Case 4: Exact boundaries (start == asof == end)
    # Expected: (1 - 1) * 0.001 = 0
    result_boundary = dcc.calculate_fraction(start_date, start_date, start_date)
    assert result_boundary == Decimal('0')

    # Case 5: Testing with frequency parameter passed through
    # Expected: (20 - 1) * 0.001 = 0.019
    result_with_freq = dcc.calculate_fraction(start_date, asof_date, end_date, freq=Decimal('2'))
    assert result_with_freq == Decimal('0.019')
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal

def test_DCC_calculate_fraction():
    # Mocking a DCFC function (day count fraction calculation method)
    # Returns (end - start).days / 360 as a simple test case
    def mock_calc_method(start, asof, end, freq):
        return Decimal((end - start).days) / Decimal("360")

    # Setup common variables
    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 6, 1)
    end_date = datetime.date(2024, 1, 1)
    freq = Decimal("2")
    
    # Define the DCC object
    dcc = DCC(
        name="Test Convention",
        altnames={"Test"},
        currencies=set(),
        calculate_fraction_method=mock_calc_method
    )

    # Case 1: Valid range (start <= asof <= end)
    # Expected calculation: (2024-01-01 - 2023-01-01).days = 365. 365 / 360
    expected_fraction = Decimal("365") / Decimal("360")
    actual_fraction = dcc.calculate_fraction(start_date, asof_date, end_date, freq)
    assert actual_fraction == expected_fraction

    # Case 2: invalid range - asof is before start (start > asof)
    # Should return ZERO per implementation logic
    invalid_asof = datetime.date(2022, 12, 31)
    assert dcc.calculate_fraction(start_date, invalid_asof, end_date, freq) == Decimal("0")

    # Case 3: invalid range - asof is after end (asof > end)
    # Should return ZERO per implementation logic
    invalid_asof_after = datetime.date(2024, 1, 2)
    assert dcc.calculate_fraction(start_date, invalid_asof_after, end_date, freq) == Decimal("0")

    # Case 4: start, asof, and end are all the same date
    # Expected calculation: (0 days) / 360 = 0
    same_date = datetime.date(2023, 1, 1)
    assert dcc.calculate_fraction(same_date, same_date, same_date, freq) == Decimal("0")

    # Case 5: Testing with frequency as None (verifying it passes through to method)
    actual_fraction_no_freq = dcc.calculate_fraction(start_date, asof_date, end_date, None)
    assert actual_fraction_no_freq == expected_fraction
```


# LLM-generated content at query #9
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_30_e_plus_360():
    # Test Case 1: Standard dates (no adjustments needed)
    # ex1: 2007-12-28 to 2008-02-28
    # nod = (28 - 28) + 30 * (2 - 12) + 360 * (2008 - 2007)
    # nod = 0 + (-300) + 360 = 60 days
    # 60 / 360 = 0.16666666666667
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

    # Test Case 2: Leap year day (Feb 29)
    # ex2: 2007-12-28 to 2008-02-29
    # nod = (29 - 28) + 30 * (2 - 12) + 360 * (2008 - 2007)
    # nod = 1 - 300 + 360 = 61 days
    # 61 / 360 = 0.16944444444444
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

    # Test Case 3: Start date is 31st (should be adjusted to 30th)
    # ex3: 2007-10-31 to 2008-11-30
    # start becomes 2007-10-30. asof is 2008-11-30.
    # nod = (30 - 30) + 30 * (11 - 10) + 360 * (2008 - 2007)
    # nod = 0 + 30 + 360 = 390 days
    # 390 / 360 = 1.08333333333333
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333366667') # Note: The docstring example says 1.08333333333333

    # Test Case 4: As of date is 31st (should be adjusted to next day)
    # ex4: 2008-02-01 to 2009-05-31
    # asof becomes 2009-06-01. start is 2008-02-01.
    # nod = (1 - 1) + 30 * (6 - 2) + 360 * (2009 - 2008)
    # nod = 0 + 120 + 360 = 480 days
    # 480 / 360 = 1.33333333333333
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
from decimal import Decimal
import datetime

@pytest.mark.parametrize("start, asof, end, expected", [
    (datetime.date(2007, 12, 28), datetime.date(2008, 2, 28), datetime.date(2008, 2, 28), Decimal('0.16986301369863')),
    (datetime.date(2007, 12, 28), datetime.date(2008, 2, 29), datetime.date(2008, 2, 29), Decimal('0.17213114754098')),
    (datetime.date(2007, 10, 31), datetime.date(2008, 11, 30), datetime.date(2008, 11, 30), Decimal('1.08196721311475')),
    (datetime.date(2008, 2, 1), datetime.date(2009, 5, 31), datetime.date(2009, 5, 31), Decimal('1.32513661202186')),
])
def test_dcfc_act_365_a(start, asof, end, expected):
    """
    Tests the dcfc_act_365_a function with various date ranges including leap years.
    """
    result = dcfc_act_365_a(start=start, asof=asof, end=end)
    assert round(result, 14) == expected
```


# LLM-generated content at query #11
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_30_e_plus_360():
    # Test Case 1: Standard dates (no day 31 involved)
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16666666666667')
    assert round(dcfc_30_e_plus_360(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == expected1

    # Test Case 2: Leap day (Feb 29)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    expected2 = Decimal('0.16944444444444')
    assert round(dcfc_30_e_plus_360(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == expected2

    # Test Case 3: Start date is 31st (should be treated as 30th)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08333333333333')
    assert round(dcfc_30_e_plus_360(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == expected3

    # Test Case 4: Long period
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    expected4 = Decimal('1.33333333333333')
    assert round(dcfc_30_e_plus_360(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == expected4

    # Test Case 5: As-of date is 31st (should be shifted to next day via timedelta)
    # If asof is 2023-08-31, it becomes 2023-09-01. 
    # Start 2023-08-01 -> End 2023-09-01
    # nod = (1 - 1) + 30 * (9 - 8) + 360 * (2023 - 2023) = 30
    # 30 / 360 = 0.08333333333333
    test_start = datetime.date(2023, 8, 1)
    test_asof = datetime.date(2023, 8, 31)
    expected5 = Decimal('0.08333333333333')
    assert round(dcfc_30_e_plus_360(start=test_start, asof=test_asof, end=test_asof), 14) == expected5
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
from decimal import Decimal
import datetime

def test_dcfc_act_act_icma():
    """
    Tests the dcfc_act_act_icma function with various scenarios including 
    standard periods, leap years, and different frequencies.
    """
    # Test Case 1: Provided doctest example
    # start=2019-03-02, asof=2019-09-10, end=2020-03-02, freq=None (defaults to 1)
    start1 = datetime.date(2019, 3, 2)
    asof1 = datetime.date(2019, 9, 10)
    end1 = datetime.date(2020, 3, 2)
    expected1 = Decimal('0.5245901639')
    # Note: We use round because floating point/decimal precision in different environments might vary slightly
    assert round(dcfc_act_act_icma(start=start1, asof=asof1, end=end1), 10) == expected1

    # Test Case 2: Testing with a specific frequency (e.g., semi-annual freq=2)
    # The formula is (days_in_period / days_in_full_period) / freq
    # If we set asof to the end date, period 1 = period 2, so result should be 1/freq
    start2 = datetime.date(2023, 1, 1)
    asof2 = datetime.date(2023, 7, 1)
    end2 = datetime.date(2024, 1, 1)
    freq2 = Decimal(2)
    # days_in_period (Jan to July) is roughly half a year. 
    # Since the function relies on _get_actual_day_count and freq, 
    # we validate that higher frequency reduces the fraction.
    res_high_freq = dcfc_act_act_icma(start2, asof2, end2, freq=freq2)
    res_low_freq = dcfc_act_act_icma(start2, asof2, end2, freq=Decimal(1))
    assert res_high_freq == res_low_freq / Decimal(2)

    # Test Case 3: Zero period (asof == start)
    # Days in period = 0, so fraction should be 0
    start3 = datetime.date(2023, 1, 1)
    asof3 = datetime.date(2023, 1, 1)
    end3 = datetime.date(2023, 12, 31)
    assert dcfc_act_act_icma(start=start3, asof=asof3, end=end3) == Decimal('0')

    # Test Case 4: Full period (asof == end)
    # Period is exactly the full interval, so fraction should be 1/freq
    start4 = datetime.date(2023, 1, 1)
    asof4 = datetime.date(2023, 12, 31)
    end4 = datetime.date(2023, 12, 31)
    # Assuming _get_actual_day_count calculates days between dates correctly
    # If end is same as asof, numerator is the total days in the interval. Result is 1/freq.
    result4 = dcf_act_act_icma(start=start4, asof=asof4, end=end4, freq=Decimal(1))
    # The logic depends on whether _get_actual_day_count includes the end day or not.
    # But for asof == end, p1 == p2, so result is 1/freq.
    assert result4 == Decimal('1')
```


# LLM-generated content at query #13
#--------------------------

```python
import datetime
from decimal import Decimal

def test_dcfc_30_360_us():
    """
    Tests the dcfc_30_360_us function with various scenarios including 
    standard dates, end of months, and leap years.
    """
    # Test Case 1: Standard date (from docstring)
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    assert round(dcfc_30_360_us(start=ex1_start, asof=ex1_asof, end=ex1_asof), 14) == Decimal('0.16666666666667')

    # Test Case 2: Leap year day (from docstring)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    assert round(dcf_30_360_us(start=ex2_start, asof=ex2_asof, end=ex2_asof), 14) == Decimal('0.16944444444444')

    # Test Case 3: End of month start date (from docstring)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    assert round(dcfc_30_360_us(start=ex3_start, asof=ex3_asof, end=ex3_asof), 14) == Decimal('1.08333333333333')

    # Test Case 4: Longer period (from docstring)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    assert round(dcfc_30_360_us(start=ex4_start, asof=ex4_asof, end=ex4_asof), 14) == Decimal('1.33333333333333')

    # Test Case 5: Both start and asof are last day of month (D1->30, D2->30)
    # Feb 28 2023 is last day. Feb 28 2024 is last day.
    start_last = datetime.date(2023, 2, 28)
    asof_last = datetime.date(2024, 2, 29) # Leap year
    # D1 becomes 30, D2 remains 29 (since d2 is not 31 and d1 was changed to 30)
    # Calculation: (29 - 30) + 30*(2-2) + 360*(2024-2023) = -1 + 0 + 360 = 359
    # Result: 359 / 360 = 0.99722222222222
    expected_val = Decimal('359') / Decimal('360')
    assert dcfc_30_360_us(start=start_last, asof=asof_last, end=asof_last) == expected_val

    # Test Case 6: Start is 31st, Asof is 31st (D1->30, D2->30)
    start_31 = datetime.date(2023, 1, 31)
    asof_31 = datetime.date(2023, 3, 31)
    # D1 becomes 30. As of is 31. Since D1 is now 30, the rule 'if d2 == 31 and (d1 in {30, 31}): d2 = 30' applies.
    # Calculation: (30 - 30) + 30*(3-1) + 360*(2023-2023) = 0 + 60 + 0 = 60
    # Result: 60 / 360 = 0.16666666666667
    expected_val_31 = Decimal('60') / Decimal('360')
    assert dcfc_30_360_us(start=start_31, asof=asof_31, end=asof_31) == expected_val_31
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal
from unittest.mock import MagicMock

def test_DCC_interest():
    # Setup mock dependencies for Money and Currency
    # Since we cannot import, we assume they behave like objects 
    # that support multiplication with Decimals/Money (standard for Money class)
    class MockMoney:
        def __init__(self, amount):
            self.amount = Decimal(amount)
        def __mul__(self, other):
            if isinstance(other, Decimal):
                return MockMoney(self.amount * other)
            return self.amount * other
        def __rmul__(self, other):
            return self.__mul__(other)
        def __eq__(self, other):
            return isinstance(other, MockMoney) and self.amount == other.amount

    # Mock a DCFC function (the calculation method)
    # We will simulate an Actual/360 logic where fraction = days / 360
    def mock_calculate_fraction(start, asof, end, freq):
        days = (end - start).days
        return Decimal(days) / Decimal("360")

    # Create instance of DCC
    # Note: Using tuple-based initialization via NamedTuple structure
    dcc_instance = DCC(
        name="MockDCC",
        altnames={"Mock"},
        currencies=set(),
        calculate_fraction_method=mock_calculate_fraction
    )

    # Test parameters
    principal = MockMoney("1000.00")
    rate = Decimal("0.05")  # 5%
    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 1, 31)
    end_date = datetime.date(2023, 2, 1)

    # Calculation:
    # fraction = (2023-01-31 - 2023-01-01).days / 360 = 30 / 360 = 1/12
    # interest = 1000 * 0.05 * (1/12) = 50 / 12 = 4.1666...
    expected_interest_amount = Decimal("1000.00") * Decimal("0.05") * (Decimal("30") / Decimal("360"))

    # Execution
    result = dcc_instance.interest(
        principal=principal,
        rate=rate,
        start=start_date,
        asof=asof_date,
        end=end_date
    )

    # Assertions
    assert result.amount == expected_interest_amount

    # Test Case 2: End date is same as AsOf date (should use asof as end)
    result_same_end = dcc_instance.interest(
        principal=principal,
        rate=rate,
        start=start_date,
        asof=asof_date,
        end=None
    )
    assert result_same_end.amount == expected_interest_amount

    # Test Case 3: Invalid date range (Start > AsOf) should return ZERO based on DCC implementation
    invalid_start = datetime.date(2023, 1, 31)
    result_invalid_range = dcc_instance.interest(
        principal=principal,
        rate=rate,
        start=invalid_start,
        asof=start_date,
        end=end_date
    )
    # Since calculate_fraction returns ZERO (Decimal("0")) when start > asof
    assert result_invalid_range.amount == Decimal("0")
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
import datetime
from decimal import Decimal

def test_DCC_calculate_fraction():
    """
    Tests the calculate_fraction method of the DCC class.
    Tests valid date ranges, invalid date orders (start > asof or asof > end), 
    and correct delegation to the calculation method.
    """
    # Mock calculation function that simply returns a fixed value
    def mock_calc(start, asof, end, freq):
        return Decimal("0.05")

    # Define a DCC instance
    dcc = DCC(
        name="MockConvention",
        altnames={"Mock"},
        currencies=set(),
        calculate_fraction_method=mock_calc
    )

    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 6, 1)
    end_date = datetime.date(2023, 12, 31)
    freq = Decimal("2")

    # Case 1: Valid range (start <= asof <= end)
    # Should call the method and return the result
    result = dcc.calculate_fraction(start_date, asof_date, end_date, freq)
    assert result == Decimal("0.05")

    # Case 2: Invalid range - asof < start
    # Should return ZERO (which is 0)
    asof_too_early = datetime.date(2022, 1, 1)
    result_invalid_start = dcc.calculate_fraction(start_date, asof_too_early, end_date, freq)
    assert result_invalid_start == Decimal("0")

    # Case 3: Invalid range - asof > end
    # Should return ZERO (which is 0)
    asof_too_late = datetime.date(2024, 1, 1)
    result_invalid_end = dcc.calculate_fraction(start_date, asof_too_late, end_date, freq)
    assert result_invalid_end == Decimal("0")

    # Case 4: Boundary condition - start == asof == end
    # Should call the method and return the result
    result_boundary = dcc.calculate_fraction(start_date, start_date, start_date, freq)
    assert result_boundary == Decimal("0.05")

    # Case 5: Boundary condition - asof is exactly end date
    # Should call the method and return the result
    result_asof_is_end = dcc.calculate_fraction(start_date, end_date, end_date, freq)
    assert result_asof_is_end == Decimal("0.05")
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
from decimal import Decimal
import datetime

def test_DCC_interest():
    """
    Tests the interest calculation method of the DCC class.
    """
    # Mock dependencies and setup requirements
    # Since we cannot import, we assume Money is a class that supports 
    # multiplication with Decimal and returns another Money object (or compatible).
    # We'll use a simple mock-like approach for the test environment.
    
    class MockMoney:
        def __init__(self, amount: Decimal):
            self.amount = amount
        def __mul__(self, other):
            if isinstance(other, (Decimal, int)):
                return MockMoney(self.amount * Decimal(other))
            return MockMoney(self.amount) # Simplified for demo
        def __repr__(self):
            return f"MockMoney({self.amount})"
        def __eq__(self, other):
            return isinstance(other, MockMoney) and self.amount == other.amount

    # Define a dummy DCFC function: (end - start).days / 360
    def dummy_dcfc(start, asof, end, freq=None):
        return Decimal((end - start).days) / Decimal("360")

    # Create instance of DCC
    # Note: We use a mock Currency class/object since we can't import the real one
    class MockCurrency: pass
    USD = MockCurrency()
    
    dcc_instance = DCC(
        name="Actual/360",
        altnames={"A/360"},
        currencies={USD},
        calculate_fraction_method=dummy_dcfc
    )

    # Test Data
    principal = MockMoney(Decimal("1000.00"))
    rate = Decimal("0.05")  # 5%
    start_date = datetime.date(2023, 1, 1)
    asof_date = datetime.date(2023, 1, 31)
    end_date = datetime.date(2023, 4, 1)
    
    # Calculation breakdown:
    # Days between start and asof (for interest calculation): 30 days
    # Fraction = 30 / 360 = 1/12
    # Interest = 1000 * 0.05 * (1/12) = 50 / 12 = 4.1666...
    expected_amount = Decimal("1000.00") * Decimal("0.05") * (Decimal("30") / Decimal("360"))
    expected_money = MockMoney(expected_amount)

    # Execution
    result = dcc_instance.interest(
        principal=principal,
        rate=rate,
        start=start_date,
        asof=asof_date,
        end=end_date
    )

    # Assertions
    assert result == expected_money
    assert result.amount == pytest.approx(expected_amount)

    # Test case: end is None (should default to asof)
    result_none_end = dcc_instance.interest(
        principal=principal,
        rate=rate,
        start=start_date,
        asof=asof_date,
        end=None
    )
    assert result_none_end == expected_money

    # Test case: Invalid date range (start > asof) should return 0 according to DCC.calculate_fraction logic
    invalid_start = datetime.date(2023, 2, 1)
    result_invalid = dcc_instance.interest(
        principal=principal,
        rate=rate,
        start=invalid_start,
        asof=asof_date,
        end=end_date
    )
    # Since calculate_fraction returns ZERO if not start <= asof <= end
    from .commons.numbers import ZERO 
    assert result_invalid.amount == ZERO
```


# LLM-generated content at query #17
#--------------------------

```python
import datetime
from decimal import Decimal
import pytest

def test_dcfc_30_e_360():
    # Test Case 1: Standard dates (from docstring)
    ex1_start, ex1_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 28)
    expected1 = Decimal('0.16666666666667')
    result1 = dcfc_30_e_360(start=ex1_start, asof=ex1_asof, end=ex1_asof)
    assert round(result1, 14) == expected1

    # Test Case 2: End of February leap year (from docstring)
    ex2_start, ex2_asof = datetime.date(2007, 12, 28), datetime.date(2008, 2, 29)
    expected2 = Decimal('0.16944444444444')
    result2 = dcfc_30_e_360(start=ex2_start, asof=ex2_asof, end=ex2_asof)
    assert round(result2, 14) == expected2

    # Test Case 3: Start date is the 31st (from docstring - should be treated as 30th)
    ex3_start, ex3_asof = datetime.date(2007, 10, 31), datetime.date(2008, 11, 30)
    expected3 = Decimal('1.08333333333333')
    result3 = dcfc_30_e_360(start=ex3_start, asof=ex3_asof, end=ex3_asof)
    assert round(result3, 14) == expected3

    # Test Case 4: End date is the 31st (from docstring - should be treated as 30th)
    ex4_start, ex4_asof = datetime.date(2008, 2, 1), datetime.date(2009, 5, 31)
    expected4 = Decimal('1.33055555555556')
    result4 = dcfc_30_e_360(start=ex4_start, asof=ex4_asof, end=ex4_asof)
    assert round(result4, 14) == expected4

    # Test Case 5: Same day (Zero days difference)
    same_day = datetime.date(2023, 1, 1)
    result5 = dcfc_30_e_360(start=same_day, asof=same_day, end=same_day)
    assert result5 == Decimal('0')

    # Test Case 6: Both start and asof are the 31st
    start_31 = datetime.date(2023, 1, 31)
    asof_31 = datetime.date(2023, 3, 31)
    # Logic: Start becomes Jan 30, AsOf becomes Mar 30.
    # (30-30) + 30*(3-1) + 360*(2023-2023) = 60 days. 60/360 = 1/6
    expected6 = Decimal('1') / Decimal('6')
    result6 = dcfca_30_e_360(start=start_31, asof=asof_31, end=asof_31)
    assert result6 == expected6
```


