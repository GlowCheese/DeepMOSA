####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_timestamp_with_posix_format():
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.POSIX)
    assert isinstance(result, int)
    assert result > 0

def test_timestamp_with_rfc_3339_format():
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.RFC_3339)
    assert isinstance(result, str)
    assert len(result) == 20
    assert result.endswith('Z')

def test_timestamp_with_iso_8601_format():
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.ISO_8601)
    assert isinstance(result, str)
    assert 'T' in result
    assert '.' in result

def test_timestamp_with_custom_datetime_kwargs():
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.POSIX, start=2020, end=2021)
    assert isinstance(result, int)
    assert result >= 1577836800
    assert result <= 1640995199

def test_timestamp_with_timezone():
    dt = Datetime()
    result = dt.timestamp(fmt=TimestampFormat.POSIX, timezone='UTC')
    assert isinstance(result, int)
    assert result > 0

def test_timestamp_default_format_is_posix():
    dt = Datetime()
    result = dt.timestamp()
    assert isinstance(result, int)
    assert result > 0

def test_timestamp_raises_error_with_invalid_timezone():
    dt = Datetime()
    try:
        dt.timestamp(timezone='Invalid/Timezone')
    except Exception as e:
        assert isinstance(e, ImportError) or isinstance(e, pytz.exceptions.UnknownTimeZoneError)


# LLM-generated content at query #2
#--------------------------

def test_duration_with_default_parameters():
    dt = Datetime()
    result = dt.duration()
    assert isinstance(result, timedelta)
    assert result >= timedelta(minutes=1)
    assert result <= timedelta(minutes=10)

def test_duration_with_custom_min_max():
    dt = Datetime()
    result = dt.duration(min_duration=5, max_duration=15)
    assert isinstance(result, timedelta)
    assert result >= timedelta(minutes=5)
    assert result <= timedelta(minutes=15)

def test_duration_with_hours_unit():
    dt = Datetime()
    result = dt.duration(duration_unit=DurationUnit.HOURS)
    assert isinstance(result, timedelta)
    assert result >= timedelta(hours=1)
    assert result <= timedelta(hours=10)

def test_duration_with_days_unit():
    dt = Datetime()
    result = dt.duration(duration_unit=DurationUnit.DAYS)
    assert isinstance(result, timedelta)
    assert result >= timedelta(days=1)
    assert result <= timedelta(days=10)

def test_duration_with_seconds_unit():
    dt = Datetime()
    result = dt.duration(duration_unit=DurationUnit.SECONDS)
    assert isinstance(result, timedelta)
    assert result >= timedelta(seconds=1)
    assert result <= timedelta(seconds=10)

def test_duration_with_microseconds_unit():
    dt = Datetime()
    result = dt.duration(duration_unit=DurationUnit.MICROSECONDS)
    assert isinstance(result, timedelta)
    assert result >= timedelta(microseconds=1)
    assert result <= timedelta(microseconds=10)

def test_duration_with_milliseconds_unit():
    dt = Datetime()
    result = dt.duration(duration_unit=DurationUnit.MILLISECONDS)
    assert isinstance(result, timedelta)
    assert result >= timedelta(milliseconds=1)
    assert result <= timedelta(milliseconds=10)

def test_duration_with_weeks_unit():
    dt = Datetime()
    result = dt.duration(duration_unit=DurationUnit.WEEKS)
    assert isinstance(result, timedelta)
    assert result >= timedelta(weeks=1)
    assert result <= timedelta(weeks=10)

def test_duration_with_none_unit():
    dt = Datetime()
    result = dt.duration(duration_unit=None)
    assert isinstance(result, timedelta)

def test_duration_with_min_greater_than_max_raises_value_error():
    dt = Datetime()
    try:
        dt.duration(min_duration=10, max_duration=5)
        assert False
    except ValueError as e:
        assert str(e) == "min_duration must be less or equal to max_duration"

def test_duration_with_non_integer_min_raises_type_error():
    dt = Datetime()
    try:
        dt.duration(min_duration=1.5, max_duration=10)
        assert False
    except TypeError as e:
        assert str(e) == "min_duration and max_duration must be integers"

def test_duration_with_non_integer_max_raises_type_error():
    dt = Datetime()
    try:
        dt.duration(min_duration=1, max_duration=10.5)
        assert False
    except TypeError as e:
        assert str(e) == "min_duration and max_duration must be integers"


# LLM-generated content at query #3
#--------------------------

def test_datetime_with_default_parameters():
    dt_provider = Datetime()
    result = dt_provider.datetime()
    assert isinstance(result, datetime)
    assert result.year == Datetime._CURRENT_YEAR

def test_datetime_with_custom_year_range():
    dt_provider = Datetime()
    start_year = 2020
    end_year = 2023
    result = dt_provider.datetime(start=start_year, end=end_year)
    assert isinstance(result, datetime)
    assert start_year <= result.year <= end_year

def test_datetime_with_timezone_without_pytz():
    dt_provider = Datetime()
    try:
        dt_provider.datetime(timezone="UTC")
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"

def test_datetime_combines_date_and_time():
    dt_provider = Datetime()
    mock_date = date(2022, 5, 15)
    mock_time = time(14, 30, 45)
    dt_provider.date = lambda start, end: mock_date
    dt_provider.time = lambda: mock_time
    result = dt_provider.datetime(start=2022, end=2022)
    assert result == datetime.combine(date=mock_date, time=mock_time)

def test_datetime_with_same_start_and_end():
    dt_provider = Datetime()
    year = 2021
    result = dt_provider.datetime(start=year, end=year)
    assert result.year == year

def test_datetime_returns_datetime_instance():
    dt_provider = Datetime()
    result = dt_provider.datetime(start=2019, end=2020)
    assert isinstance(result, datetime)


# LLM-generated content at query #4
#--------------------------

def test_datetime_raises_import_error_when_timezone_given_and_pytz_not_installed():
    mock_pytz = None
    import sys
    sys.modules['pytz'] = mock_pytz
    dt_provider = Datetime()
    try:
        dt_provider.datetime(timezone="UTC")
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #5
#--------------------------

def test_bulk_create_datetimes_raises_value_error_when_no_dates_provided():
    from mimesis import Datetime
    dt = Datetime()
    try:
        dt.bulk_create_datetimes(None, None)
    except ValueError as e:
        assert str(e) == "You must pass date_start and date_end"

def test_bulk_create_datetimes_raises_value_error_when_date_start_greater_than_date_end():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    start = datetime(2023, 1, 1)
    end = datetime(2022, 1, 1)
    try:
        dt.bulk_create_datetimes(start, end)
    except ValueError as e:
        assert str(e) == "date_start can not be larger than date_end"

def test_bulk_create_datetimes_raises_value_error_when_timedelta_not_positive():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 2)
    try:
        dt.bulk_create_datetimes(start, end, days=0)
    except ValueError as e:
        assert str(e) == "timedelta must be positive"

def test_bulk_create_datetimes_creates_list_with_correct_step():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 3)
    result = dt.bulk_create_datetimes(start, end, days=1)
    expected = [datetime(2023, 1, 2), datetime(2023, 1, 3)]
    assert result == expected

def test_bulk_create_datetimes_creates_list_with_hours_step():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    start = datetime(2023, 1, 1, 0, 0, 0)
    end = datetime(2023, 1, 1, 2, 0, 0)
    result = dt.bulk_create_datetimes(start, end, hours=1)
    expected = [datetime(2023, 1, 1, 1, 0, 0), datetime(2023, 1, 1, 2, 0, 0)]
    assert result == expected

def test_bulk_create_datetimes_returns_empty_list_when_start_equals_end_and_step_positive():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 1)
    result = dt.bulk_create_datetimes(start, end, days=1)
    assert result == []

def test_bulk_create_datetimes_handles_microseconds_step():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    start = datetime(2023, 1, 1, 0, 0, 0, 0)
    end = datetime(2023, 1, 1, 0, 0, 0, 2000)
    result = dt.bulk_create_datetimes(start, end, microseconds=1000)
    expected = [datetime(2023, 1, 1, 0, 0, 0, 1000), datetime(2023, 1, 1, 0, 0, 0, 2000)]
    assert result == expected

def test_bulk_create_datetimes_works_with_combined_timedelta_arguments():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    start = datetime(2023, 1, 1, 0, 0, 0)
    end = datetime(2023, 1, 2, 1, 1, 1)
    result = dt.bulk_create_datetimes(start, end, days=1, hours=1, minutes=1, seconds=1)
    expected = [datetime(2023, 1, 2, 1, 1, 1)]
    assert result == expected


# LLM-generated content at query #6
#--------------------------

def test_datetime_timezone_without_pytz():
    import sys
    from unittest.mock import patch

    from mimesis import Datetime
    dt = Datetime()
    with patch.dict(sys.modules, {'pytz': None}):
        try:
            dt.datetime(timezone='UTC')
        except ImportError as e:
            assert str(e) == 'Timezones are supported only with pytz'


# LLM-generated content at query #7
#--------------------------

def test_bulk_create_datetimes_valid_range_and_step():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    expected = [datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 4), datetime(2023, 1, 5), datetime(2023, 1, 6)]
    assert result == expected

def test_bulk_create_datetimes_empty_start_and_end():
    try:
        Datetime.bulk_create_datetimes(None, None)
        assert False
    except ValueError as e:
        assert str(e) == "You must pass date_start and date_end"

def test_bulk_create_datetimes_start_larger_than_end():
    date_start = datetime(2023, 1, 5)
    date_end = datetime(2023, 1, 1)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=1)
        assert False
    except ValueError as e:
        assert str(e) == "date_start can not be larger than date_end"

def test_bulk_create_datetimes_non_positive_timedelta():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=0)
        assert False
    except ValueError as e:
        assert str(e) == "timedelta must be positive"

def test_bulk_create_datetimes_zero_timedelta():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end)
        assert False
    except ValueError as e:
        assert str(e) == "timedelta must be positive"

def test_bulk_create_datetimes_hours_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 4, 0, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, hours=1)
    expected = [datetime(2023, 1, 1, 1, 0, 0), datetime(2023, 1, 1, 2, 0, 0), datetime(2023, 1, 1, 3, 0, 0), datetime(2023, 1, 1, 4, 0, 0), datetime(2023, 1, 1, 5, 0, 0)]
    assert result == expected

def test_bulk_create_datetimes_minutes_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 2, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, minutes=1)
    expected = [datetime(2023, 1, 1, 0, 1, 0), datetime(2023, 1, 1, 0, 2, 0), datetime(2023, 1, 1, 0, 3, 0)]
    assert result == expected

def test_bulk_create_datetimes_seconds_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 0, 2)
    result = Datetime.bulk_create_datetimes(date_start, date_end, seconds=1)
    expected = [datetime(2023, 1, 1, 0, 0, 1), datetime(2023, 1, 1, 0, 0, 2), datetime(2023, 1, 1, 0, 0, 3)]
    assert result == expected

def test_bulk_create_datetimes_microseconds_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 0, 0, 2000)
    result = Datetime.bulk_create_datetimes(date_start, date_end, microseconds=1000)
    expected = [datetime(2023, 1, 1, 0, 0, 0, 1000), datetime(2023, 1, 1, 0, 0, 0, 2000), datetime(2023, 1, 1, 0, 0, 0, 3000)]
    assert result == expected

def test_bulk_create_datetimes_combined_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 1, 30, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, hours=1, minutes=30)
    expected = [datetime(2023, 1, 1, 1, 30, 0), datetime(2023, 1, 1, 3, 0, 0)]
    assert result == expected

def test_bulk_create_datetimes_exact_match():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 1)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    expected = [datetime(2023, 1, 2)]
    assert result == expected


# LLM-generated content at query #8
#--------------------------

def test_bulk_create_datetimes_valid_range_and_step():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    expected = [datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 4), datetime(2023, 1, 5), datetime(2023, 1, 6)]
    assert result == expected

def test_bulk_create_datetimes_empty_date_start_and_date_end():
    try:
        Datetime.bulk_create_datetimes(None, None)
        assert False
    except ValueError as e:
        assert str(e) == "You must pass date_start and date_end"

def test_bulk_create_datetimes_date_start_larger_than_date_end():
    date_start = datetime(2023, 1, 5)
    date_end = datetime(2023, 1, 1)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=1)
        assert False
    except ValueError as e:
        assert str(e) == "date_start can not be larger than date_end"

def test_bulk_create_datetimes_non_positive_timedelta():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=0)
        assert False
    except ValueError as e:
        assert str(e) == "timedelta must be positive"

def test_bulk_create_datetimes_with_hours_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 6, 0, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, hours=2)
    expected = [datetime(2023, 1, 1, 2, 0, 0), datetime(2023, 1, 1, 4, 0, 0), datetime(2023, 1, 1, 6, 0, 0), datetime(2023, 1, 1, 8, 0, 0)]
    assert result == expected

def test_bulk_create_datetimes_single_step_exact_end():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 2)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    expected = [datetime(2023, 1, 2), datetime(2023, 1, 3)]
    assert result == expected

def test_bulk_create_datetimes_microseconds_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 0, 0, 2000)
    result = Datetime.bulk_create_datetimes(date_start, date_end, microseconds=500)
    expected = [datetime(2023, 1, 1, 0, 0, 0, 500), datetime(2023, 1, 1, 0, 0, 0, 1000), datetime(2023, 1, 1, 0, 0, 0, 1500), datetime(2023, 1, 1, 0, 0, 0, 2000), datetime(2023, 1, 1, 0, 0, 0, 2500)]
    assert result == expected


# LLM-generated content at query #9
#--------------------------

def test_datetime_with_default_parameters():
    dt = Datetime()
    result = dt.datetime()
    assert isinstance(result, datetime)
    assert result.year == Datetime._CURRENT_YEAR

def test_datetime_with_custom_year_range():
    dt = Datetime()
    result = dt.datetime(start=2020, end=2023)
    assert isinstance(result, datetime)
    assert 2020 <= result.year <= 2023

def test_datetime_with_timezone_without_pytz():
    dt = Datetime()
    try:
        dt.datetime(timezone="UTC")
        assert False
    except ImportError as e:
        assert "Timezones are supported only with pytz" in str(e)

def test_datetime_with_timezone_with_pytz(mocker):
    mocker.patch('pytz.timezone')
    dt = Datetime()
    result = dt.datetime(timezone="America/New_York")
    assert isinstance(result, datetime)

def test_datetime_combines_date_and_time():
    dt = Datetime()
    mock_date = date(2022, 5, 15)
    mock_time = time(14, 30, 45)
    mocker.patch.object(dt, 'date', return_value=mock_date)
    mocker.patch.object(dt, 'time', return_value=mock_time)
    result = dt.datetime()
    assert result.date() == mock_date
    assert result.time() == mock_time

def test_datetime_start_greater_than_end():
    dt = Datetime()
    result = dt.datetime(start=2023, end=2020)
    assert isinstance(result, datetime)
    assert 2020 <= result.year <= 2023


# LLM-generated content at query #10
#--------------------------

def test_bulk_create_datetimes_with_positive_timedelta():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    kwargs = {"days": 1}
    result = dt.bulk_create_datetimes(date_start, date_end, **kwargs)
    assert isinstance(result, list)
    assert all(isinstance(item, datetime) for item in result)
    assert len(result) > 0


# LLM-generated content at query #11
#--------------------------

def test_datetime_timezone_import_error():
    dt = Datetime()
    try:
        dt.datetime(timezone="UTC")
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #12
#--------------------------

def test_bulk_create_datetimes_positive_timedelta():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    result = dt.bulk_create_datetimes(date_start, date_end, days=1)
    assert isinstance(result, list)
    assert all(isinstance(item, datetime) for item in result)
    assert len(result) > 0


# LLM-generated content at query #13
#--------------------------

def test_bulk_create_datetimes_valid_range_and_step():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    expected = [datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 4), datetime(2023, 1, 5), datetime(2023, 1, 6)]
    assert result == expected

def test_bulk_create_datetimes_empty_start_and_end():
    try:
        Datetime.bulk_create_datetimes(None, None)
        assert False
    except ValueError as e:
        assert str(e) == "You must pass date_start and date_end"

def test_bulk_create_datetimes_start_larger_than_end():
    date_start = datetime(2023, 1, 5)
    date_end = datetime(2023, 1, 1)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=1)
        assert False
    except ValueError as e:
        assert str(e) == "date_start can not be larger than date_end"

def test_bulk_create_datetimes_non_positive_timedelta():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=0)
        assert False
    except ValueError as e:
        assert str(e) == "timedelta must be positive"

def test_bulk_create_datetimes_negative_timedelta():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=-1)
        assert False
    except ValueError as e:
        assert str(e) == "timedelta must be positive"

def test_bulk_create_datetimes_hours_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 4, 0, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, hours=1)
    expected = [datetime(2023, 1, 1, 1, 0, 0), datetime(2023, 1, 1, 2, 0, 0), datetime(2023, 1, 1, 3, 0, 0), datetime(2023, 1, 1, 4, 0, 0), datetime(2023, 1, 1, 5, 0, 0)]
    assert result == expected

def test_bulk_create_datetimes_minutes_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 2, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, minutes=1)
    expected = [datetime(2023, 1, 1, 0, 1, 0), datetime(2023, 1, 1, 0, 2, 0), datetime(2023, 1, 1, 0, 3, 0)]
    assert result == expected

def test_bulk_create_datetimes_seconds_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 0, 2)
    result = Datetime.bulk_create_datetimes(date_start, date_end, seconds=1)
    expected = [datetime(2023, 1, 1, 0, 0, 1), datetime(2023, 1, 1, 0, 0, 2), datetime(2023, 1, 1, 0, 0, 3)]
    assert result == expected

def test_bulk_create_datetimes_microseconds_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 0, 0, 2000)
    result = Datetime.bulk_create_datetimes(date_start, date_end, microseconds=1000)
    expected = [datetime(2023, 1, 1, 0, 0, 0, 1000), datetime(2023, 1, 1, 0, 0, 0, 2000), datetime(2023, 1, 1, 0, 0, 0, 3000)]
    assert result == expected

def test_bulk_create_datetimes_combined_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 1, 30, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, hours=1, minutes=30)
    expected = [datetime(2023, 1, 1, 1, 30, 0), datetime(2023, 1, 1, 3, 0, 0)]
    assert result == expected

def test_bulk_create_datetimes_single_step_exact_end():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 2)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    expected = [datetime(2023, 1, 2), datetime(2023, 1, 3)]
    assert result == expected

def test_bulk_create_datetimes_step_larger_than_range():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 3)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=5)
    expected = [datetime(2023, 1, 6)]
    assert result == expected


# LLM-generated content at query #14
#--------------------------

def test_datetime_timezone_without_pytz():
    import sys
    from unittest.mock import patch

    from mimesis.providers.datetime import Datetime
    dt = Datetime()
    with patch.dict(sys.modules, {'pytz': None}):
        try:
            dt.datetime(timezone='UTC')
        except ImportError as e:
            assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #15
#--------------------------

def test_datetime_timezone_without_pytz():
    mock_pytz = None
    import sys
    sys.modules['pytz'] = mock_pytz
    dt_provider = Datetime()
    try:
        dt_provider.datetime(timezone="UTC")
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #16
#--------------------------

def test_bulk_create_datetimes_positive_timedelta():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    result = dt.bulk_create_datetimes(date_start, date_end, days=1)
    assert isinstance(result, list)
    assert all(isinstance(item, datetime) for item in result)
    assert len(result) > 0


# LLM-generated content at query #17
#--------------------------

def test_bulk_create_datetimes_with_positive_timedelta():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    result = dt.bulk_create_datetimes(date_start, date_end, days=1)
    assert isinstance(result, list)
    assert all(isinstance(item, datetime) for item in result)
    assert len(result) > 0


# LLM-generated content at query #18
#--------------------------

def test_datetime_with_default_parameters():
    dt = Datetime()
    result = dt.datetime()
    assert isinstance(result, datetime)
    assert result.year == Datetime._CURRENT_YEAR

def test_datetime_with_custom_year_range():
    dt = Datetime()
    start_year = 2010
    end_year = 2020
    result = dt.datetime(start=start_year, end=end_year)
    assert isinstance(result, datetime)
    assert start_year <= result.year <= end_year

def test_datetime_with_timezone_without_pytz():
    dt = Datetime()
    try:
        dt.datetime(timezone="UTC")
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"

def test_datetime_with_timezone_with_pytz(mocker):
    mocker.patch('pytz.timezone')
    dt = Datetime()
    result = dt.datetime(timezone="America/New_York")
    assert isinstance(result, datetime)

def test_datetime_combines_date_and_time():
    dt = Datetime()
    mock_date = date(2020, 5, 15)
    mock_time = time(14, 30, 45)
    mocker.patch.object(dt, 'date', return_value=mock_date)
    mocker.patch.object(dt, 'time', return_value=mock_time)
    result = dt.datetime()
    assert result.date() == mock_date
    assert result.time() == mock_time

def test_datetime_start_greater_than_end():
    dt = Datetime()
    result = dt.datetime(start=2020, end=2010)
    assert isinstance(result, datetime)
    assert 2010 <= result.year <= 2020

def test_datetime_same_start_and_end():
    dt = Datetime()
    year = 2015
    result = dt.datetime(start=year, end=year)
    assert isinstance(result, datetime)
    assert result.year == year

def test_datetime_negative_years():
    dt = Datetime()
    result = dt.datetime(start=-100, end=-50)
    assert isinstance(result, datetime)
    assert -100 <= result.year <= -50

def test_datetime_large_year_range():
    dt = Datetime()
    result = dt.datetime(start=1, end=9999)
    assert isinstance(result, datetime)
    assert 1 <= result.year <= 9999

def test_datetime_timezone_none():
    dt = Datetime()
    result = dt.datetime(timezone=None)
    assert isinstance(result, datetime)
    assert result.tzinfo is None

def test_datetime_timezone_empty_string():
    dt = Datetime()
    result = dt.datetime(timezone="")
    assert isinstance(result, datetime)
    assert result.tzinfo is None


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_datetime_with_default_parameters():
    dt_provider = Datetime()
    result = dt_provider.datetime()
    assert isinstance(result, datetime)
    assert result.year == Datetime._CURRENT_YEAR

def test_datetime_with_custom_year_range():
    dt_provider = Datetime()
    start_year = 2010
    end_year = 2020
    result = dt_provider.datetime(start=start_year, end=end_year)
    assert isinstance(result, datetime)
    assert start_year <= result.year <= end_year

def test_datetime_with_timezone_without_pytz():
    dt_provider = Datetime()
    try:
        dt_provider.datetime(timezone="UTC")
    except ImportError as e:
        assert "Timezones are supported only with pytz" in str(e)

def test_datetime_with_timezone_with_pytz(mocker):
    mocker.patch('mimesis.providers.datetime.pytz', mocker.MagicMock())
    dt_provider = Datetime()
    mock_tz = mocker.MagicMock()
    mocker.patch('mimesis.providers.datetime.pytz.timezone', return_value=mock_tz)
    mock_tz.localize = mocker.MagicMock(return_value=datetime(2023, 1, 1, 12, 0, 0))
    result = dt_provider.datetime(timezone="America/New_York")
    assert isinstance(result, datetime)

def test_datetime_combines_date_and_time():
    dt_provider = Datetime()
    mock_date = date(2023, 5, 15)
    mock_time = time(14, 30, 45)
    mocker.patch.object(dt_provider, 'date', return_value=mock_date)
    mocker.patch.object(dt_provider, 'time', return_value=mock_time)
    result = dt_provider.datetime()
    assert result.date() == mock_date
    assert result.time() == mock_time

def test_datetime_with_start_greater_than_end():
    dt_provider = Datetime()
    result = dt_provider.datetime(start=2023, end=2020)
    assert isinstance(result, datetime)
    assert 2020 <= result.year <= 2023

def test_datetime_returns_correct_type():
    dt_provider = Datetime()
    result = dt_provider.datetime()
    assert type(result) == datetime

def test_datetime_randomness():
    dt_provider = Datetime()
    results = [dt_provider.datetime(2020, 2020) for _ in range(10)]
    unique_results = set(results)
    assert len(unique_results) > 1

def test_datetime_with_same_start_and_end():
    dt_provider = Datetime()
    year = 2015
    result = dt_provider.datetime(start=year, end=year)
    assert result.year == year

def test_datetime_timezone_parameter_none():
    dt_provider = Datetime()
    result = dt_provider.datetime(timezone=None)
    assert isinstance(result, datetime)
    assert result.tzinfo is None


# LLM-generated content at query #2
#--------------------------

def test_duration_with_default_parameters():
    dt = Datetime()
    result = dt.duration()
    assert isinstance(result, timedelta)
    assert result.total_seconds() >= 60
    assert result.total_seconds() <= 600

def test_duration_with_custom_min_max():
    dt = Datetime()
    result = dt.duration(min_duration=5, max_duration=15)
    assert isinstance(result, timedelta)
    assert result.total_seconds() >= 300
    assert result.total_seconds() <= 900

def test_duration_with_different_units():
    dt = Datetime()
    result_seconds = dt.duration(duration_unit=DurationUnit.SECONDS)
    assert isinstance(result_seconds, timedelta)
    result_hours = dt.duration(duration_unit=DurationUnit.HOURS)
    assert isinstance(result_hours, timedelta)
    result_days = dt.duration(duration_unit=DurationUnit.DAYS)
    assert isinstance(result_days, timedelta)
    result_weeks = dt.duration(duration_unit=DurationUnit.WEEKS)
    assert isinstance(result_weeks, timedelta)

def test_duration_with_none_unit():
    dt = Datetime()
    result = dt.duration(duration_unit=None)
    assert isinstance(result, timedelta)

def test_duration_min_greater_than_max_raises_value_error():
    dt = Datetime()
    try:
        dt.duration(min_duration=10, max_duration=5)
        assert False
    except ValueError as e:
        assert str(e) == "min_duration must be less or equal to max_duration"

def test_duration_non_integer_min_raises_type_error():
    dt = Datetime()
    try:
        dt.duration(min_duration=1.5, max_duration=10)
        assert False
    except TypeError as e:
        assert str(e) == "min_duration and max_duration must be integers"

def test_duration_non_integer_max_raises_type_error():
    dt = Datetime()
    try:
        dt.duration(min_duration=1, max_duration=10.5)
        assert False
    except TypeError as e:
        assert str(e) == "min_duration and max_duration must be integers"


# LLM-generated content at query #3
#--------------------------

def test_datetime_raises_import_error_when_timezone_given_and_pytz_not_installed():
    dt_provider = Datetime()
    mock_pytz = None
    import sys
    original_pytz = sys.modules.get('pytz')
    sys.modules['pytz'] = mock_pytz
    try:
        dt_provider.datetime(timezone='UTC')
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"
    finally:
        if original_pytz is not None:
            sys.modules['pytz'] = original_pytz
        else:
            del sys.modules['pytz']


# LLM-generated content at query #4
#--------------------------

def test_bulk_create_datetimes_valid_range_and_step():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 3)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    expected = [datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 4)]
    assert result == expected

def test_bulk_create_datetimes_empty_arguments():
    try:
        Datetime.bulk_create_datetimes(None, None)
        assert False
    except ValueError as e:
        assert str(e) == "You must pass date_start and date_end"

def test_bulk_create_datetimes_start_larger_than_end():
    date_start = datetime(2023, 1, 3)
    date_end = datetime(2023, 1, 1)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=1)
        assert False
    except ValueError as e:
        assert str(e) == "date_start can not be larger than date_end"

def test_bulk_create_datetimes_non_positive_timedelta():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 3)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=0)
        assert False
    except ValueError as e:
        assert str(e) == "timedelta must be positive"

def test_bulk_create_datetimes_with_hours_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 4, 0, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, hours=1)
    expected = [datetime(2023, 1, 1, 1, 0, 0), datetime(2023, 1, 1, 2, 0, 0), datetime(2023, 1, 1, 3, 0, 0), datetime(2023, 1, 1, 4, 0, 0), datetime(2023, 1, 1, 5, 0, 0)]
    assert result == expected

def test_bulk_create_datetimes_single_step_exact_end():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 2)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    expected = [datetime(2023, 1, 2), datetime(2023, 1, 3)]
    assert result == expected

def test_bulk_create_datetimes_with_microseconds():
    date_start = datetime(2023, 1, 1, 0, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 0, 0, 2000)
    result = Datetime.bulk_create_datetimes(date_start, date_end, microseconds=500)
    expected = [datetime(2023, 1, 1, 0, 0, 0, 500), datetime(2023, 1, 1, 0, 0, 0, 1000), datetime(2023, 1, 1, 0, 0, 0, 1500), datetime(2023, 1, 1, 0, 0, 0, 2000), datetime(2023, 1, 1, 0, 0, 0, 2500)]
    assert result == expected

def test_bulk_create_datetimes_negative_timedelta():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 3)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=-1)
        assert False
    except ValueError as e:
        assert str(e) == "timedelta must be positive"

def test_bulk_create_datetimes_zero_timedelta():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 3)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, seconds=0)
        assert False
    except ValueError as e:
        assert str(e) == "timedelta must be positive"

def test_bulk_create_datetimes_combined_timedelta():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 2, 30, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, hours=1, minutes=30)
    expected = [datetime(2023, 1, 1, 1, 30, 0), datetime(2023, 1, 1, 3, 0, 0)]
    assert result == expected


# LLM-generated content at query #5
#--------------------------

def test_datetime_with_default_parameters():
    dt = Datetime()
    result = dt.datetime()
    assert isinstance(result, datetime)
    assert result.year == Datetime._CURRENT_YEAR

def test_datetime_with_custom_year_range():
    dt = Datetime()
    start_year = 2010
    end_year = 2020
    result = dt.datetime(start=start_year, end=end_year)
    assert isinstance(result, datetime)
    assert start_year <= result.year <= end_year

def test_datetime_with_timezone_without_pytz():
    dt = Datetime()
    try:
        dt.datetime(timezone="UTC")
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"

def test_datetime_with_timezone_with_pytz(mocker):
    mocker.patch('mimesis.providers.datetime.pytz', mocker.MagicMock())
    dt = Datetime()
    mock_tz = mocker.MagicMock()
    mocker.patch('mimesis.providers.datetime.pytz.timezone', return_value=mock_tz)
    mock_tz.localize.return_value = datetime(2023, 1, 1, 12, 0, 0)
    result = dt.datetime(timezone="America/New_York")
    assert isinstance(result, datetime)
    mock_tz.localize.assert_called_once()

def test_datetime_combines_date_and_time():
    dt = Datetime()
    mock_date = date(2023, 5, 15)
    mock_time = time(14, 30, 45)
    mocker.patch.object(dt, 'date', return_value=mock_date)
    mocker.patch.object(dt, 'time', return_value=mock_time)
    result = dt.datetime()
    assert result == datetime.combine(mock_date, mock_time)

def test_datetime_with_start_greater_than_end():
    dt = Datetime()
    result = dt.datetime(start=2023, end=2022)
    assert result.year == 2023

def test_datetime_returns_correct_type():
    dt = Datetime()
    result = dt.datetime()
    assert type(result) is datetime


# LLM-generated content at query #6
#--------------------------

def test_bulk_create_datetimes_with_positive_timedelta():
    from datetime import datetime

    from mimesis import Datetime
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(dt, datetime) for dt in result)


# LLM-generated content at query #7
#--------------------------

def test_datetime_timezone_import_error():
    dt = Datetime()
    try:
        dt.datetime(timezone="UTC")
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #8
#--------------------------

def test_bulk_create_datetimes_with_positive_timedelta():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    result = dt.bulk_create_datetimes(date_start, date_end, days=1)
    assert isinstance(result, list)
    assert all(isinstance(item, datetime) for item in result)
    assert len(result) > 0


# LLM-generated content at query #9
#--------------------------

def test_bulk_create_datetimes_with_positive_timedelta():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    result = dt.bulk_create_datetimes(date_start, date_end, days=1)
    assert isinstance(result, list)
    assert all(isinstance(item, datetime) for item in result)
    assert len(result) > 0


# LLM-generated content at query #10
#--------------------------

def test_datetime_timezone_without_pytz():
    import sys
    from unittest.mock import patch

    from mimesis.providers.datetime import Datetime
    dt = Datetime()
    with patch.dict(sys.modules, {'pytz': None}):
        try:
            dt.datetime(timezone='UTC')
        except ImportError as e:
            assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #11
#--------------------------

def test_bulk_create_datetimes_with_positive_timedelta():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    result = dt.bulk_create_datetimes(date_start, date_end, days=1)
    assert len(result) > 0
    assert all(isinstance(d, datetime) for d in result)


# LLM-generated content at query #12
#--------------------------

def test_datetime_raises_import_error_when_timezone_given_and_pytz_not_installed():
    dt_provider = Datetime()
    mock_pytz = None
    import sys
    original_pytz = sys.modules.get('pytz')
    sys.modules['pytz'] = mock_pytz
    try:
        dt_provider.datetime(timezone='UTC')
    except ImportError as e:
        assert str(e) == 'Timezones are supported only with pytz'
    finally:
        if original_pytz is None:
            del sys.modules['pytz']
        else:
            sys.modules['pytz'] = original_pytz


# LLM-generated content at query #13
#--------------------------

def test_bulk_create_datetimes_positive_timedelta():
    from datetime import datetime

    from mimesis import Datetime
    dt = Datetime()
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    result = dt.bulk_create_datetimes(date_start, date_end, days=1)
    assert isinstance(result, list)
    assert all(isinstance(item, datetime) for item in result)
    assert len(result) > 0


# LLM-generated content at query #14
#--------------------------

def test_datetime_with_default_parameters():
    dt_provider = Datetime()
    result = dt_provider.datetime()
    assert isinstance(result, datetime)
    assert result.year == Datetime._CURRENT_YEAR

def test_datetime_with_custom_year_range():
    dt_provider = Datetime()
    start_year = 2010
    end_year = 2020
    result = dt_provider.datetime(start=start_year, end=end_year)
    assert isinstance(result, datetime)
    assert start_year <= result.year <= end_year

def test_datetime_with_timezone_without_pytz():
    dt_provider = Datetime()
    try:
        dt_provider.datetime(timezone="UTC")
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"

def test_datetime_with_timezone_with_pytz(mocker):
    mocker.patch('mimesis.providers.datetime.pytz', mocker.MagicMock())
    dt_provider = Datetime()
    mock_tz = mocker.MagicMock()
    mocker.patch('mimesis.providers.datetime.pytz.timezone', return_value=mock_tz)
    mock_tz.localize.return_value = datetime(2023, 1, 1, 12, 0, 0)
    result = dt_provider.datetime(timezone="America/New_York")
    assert isinstance(result, datetime)

def test_datetime_combines_date_and_time():
    dt_provider = Datetime()
    mocker.patch.object(dt_provider, 'date', return_value=date(2023, 5, 15))
    mocker.patch.object(dt_provider, 'time', return_value=time(14, 30, 45))
    result = dt_provider.datetime()
    assert result.date() == date(2023, 5, 15)
    assert result.time() == time(14, 30, 45)

def test_datetime_with_same_start_and_end():
    dt_provider = Datetime()
    year = 2015
    result = dt_provider.datetime(start=year, end=year)
    assert result.year == year

def test_datetime_returns_valid_datetime_object():
    dt_provider = Datetime()
    result = dt_provider.datetime(start=2000, end=2020)
    assert isinstance(result, datetime)
    assert 2000 <= result.year <= 2020
    assert 1 <= result.month <= 12
    assert 1 <= result.day <= 31
    assert 0 <= result.hour <= 23
    assert 0 <= result.minute <= 59
    assert 0 <= result.second <= 59
    assert 0 <= result.microsecond <= 999999


# LLM-generated content at query #15
#--------------------------

def test_bulk_create_datetimes_valid_input():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    assert len(result) == 10
    assert result[0] == datetime(2023, 1, 2)
    assert result[-1] == datetime(2023, 1, 11)

def test_bulk_create_datetimes_empty_dates():
    try:
        Datetime.bulk_create_datetimes(None, None)
        assert False
    except ValueError as e:
        assert str(e) == "You must pass date_start and date_end"

def test_bulk_create_datetimes_start_larger_than_end():
    date_start = datetime(2023, 1, 10)
    date_end = datetime(2023, 1, 1)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=1)
        assert False
    except ValueError as e:
        assert str(e) == "date_start can not be larger than date_end"

def test_bulk_create_datetimes_non_positive_timedelta():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 10)
    try:
        Datetime.bulk_create_datetimes(date_start, date_end, days=0)
        assert False
    except ValueError as e:
        assert str(e) == "timedelta must be positive"

def test_bulk_create_datetimes_hours_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 5, 0, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, hours=1)
    assert len(result) == 5
    assert result[0] == datetime(2023, 1, 1, 1, 0, 0)
    assert result[-1] == datetime(2023, 1, 1, 5, 0, 0)

def test_bulk_create_datetimes_minutes_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 10, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, minutes=2)
    assert len(result) == 5
    assert result[0] == datetime(2023, 1, 1, 0, 2, 0)
    assert result[-1] == datetime(2023, 1, 1, 0, 10, 0)

def test_bulk_create_datetimes_seconds_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 0, 10)
    result = Datetime.bulk_create_datetimes(date_start, date_end, seconds=2)
    assert len(result) == 5
    assert result[0] == datetime(2023, 1, 1, 0, 0, 2)
    assert result[-1] == datetime(2023, 1, 1, 0, 0, 10)

def test_bulk_create_datetimes_microseconds_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0, 0)
    date_end = datetime(2023, 1, 1, 0, 0, 0, 1000)
    result = Datetime.bulk_create_datetimes(date_start, date_end, microseconds=200)
    assert len(result) == 5
    assert result[0] == datetime(2023, 1, 1, 0, 0, 0, 200)
    assert result[-1] == datetime(2023, 1, 1, 0, 0, 0, 1000)

def test_bulk_create_datetimes_combined_step():
    date_start = datetime(2023, 1, 1, 0, 0, 0)
    date_end = datetime(2023, 1, 2, 12, 0, 0)
    result = Datetime.bulk_create_datetimes(date_start, date_end, hours=12)
    assert len(result) == 3
    assert result[0] == datetime(2023, 1, 1, 12, 0, 0)
    assert result[-1] == datetime(2023, 1, 2, 12, 0, 0)

def test_bulk_create_datetimes_exact_end():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 5)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    assert len(result) == 5
    assert result[-1] == datetime(2023, 1, 6)

def test_bulk_create_datetimes_single_step():
    date_start = datetime(2023, 1, 1)
    date_end = datetime(2023, 1, 1)
    result = Datetime.bulk_create_datetimes(date_start, date_end, days=1)
    assert len(result) == 1
    assert result[0] == datetime(2023, 1, 2)


# LLM-generated content at query #16
#--------------------------

def test_datetime_timezone_import_error():
    dt = Datetime()
    try:
        dt.datetime(timezone="UTC")
    except ImportError as e:
        assert str(e) == "Timezones are supported only with pytz"


# LLM-generated content at query #17
#--------------------------

def test_bulk_create_datetimes_valid_range_and_step():
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 5)
    result = Datetime.bulk_create_datetimes(start, end, days=1)
    expected = [datetime(2023, 1, 2), datetime(2023, 1, 3), datetime(2023, 1, 4), datetime(2023, 1, 5), datetime(2023, 1, 6)]
    assert result == expected

def test_bulk_create_datetimes_empty_start_and_end_raises_value_error():
    start = None
    end = None
    try:
        Datetime.bulk_create_datetimes(start, end, days=1)
        assert False
    except ValueError as e:
        assert str(e) == "You must pass date_start and date_end"

def test_bulk_create_datetimes_start_greater_than_end_raises_value_error():
    start = datetime(2023, 1, 5)
    end = datetime(2023, 1, 1)
    try:
        Datetime.bulk_create_datetimes(start, end, days=1)
        assert False
    except ValueError as e:
        assert str(e) == "date_start can not be larger than date_end"

def test_bulk_create_datetimes_non_positive_timedelta_raises_value_error():
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 5)
    try:
        Datetime.bulk_create_datetimes(start, end, days=0)
        assert False
    except ValueError as e:
        assert str(e) == "timedelta must be positive"

def test_bulk_create_datetimes_with_hours_step():
    start = datetime(2023, 1, 1, 0, 0, 0)
    end = datetime(2023, 1, 1, 6, 0, 0)
    result = Datetime.bulk_create_datetimes(start, end, hours=2)
    expected = [datetime(2023, 1, 1, 2, 0, 0), datetime(2023, 1, 1, 4, 0, 0), datetime(2023, 1, 1, 6, 0, 0), datetime(2023, 1, 1, 8, 0, 0)]
    assert result == expected

def test_bulk_create_datetimes_with_minutes_step():
    start = datetime(2023, 1, 1, 0, 0, 0)
    end = datetime(2023, 1, 1, 0, 10, 0)
    result = Datetime.bulk_create_datetimes(start, end, minutes=3)
    expected = [datetime(2023, 1, 1, 0, 3, 0), datetime(2023, 1, 1, 0, 6, 0), datetime(2023, 1, 1, 0, 9, 0), datetime(2023, 1, 1, 0, 12, 0)]
    assert result == expected

def test_bulk_create_datetimes_with_seconds_step():
    start = datetime(2023, 1, 1, 0, 0, 0)
    end = datetime(2023, 1, 1, 0, 0, 12)
    result = Datetime.bulk_create_datetimes(start, end, seconds=4)
    expected = [datetime(2023, 1, 1, 0, 0, 4), datetime(2023, 1, 1, 0, 0, 8), datetime(2023, 1, 1, 0, 0, 12), datetime(2023, 1, 1, 0, 0, 16)]
    assert result == expected

def test_bulk_create_datetimes_with_microseconds_step():
    start = datetime(2023, 1, 1, 0, 0, 0, 0)
    end = datetime(2023, 1, 1, 0, 0, 0, 300000)
    result = Datetime.bulk_create_datetimes(start, end, microseconds=100000)
    expected = [datetime(2023, 1, 1, 0, 0, 0, 100000), datetime(2023, 1, 1, 0, 0, 0, 200000), datetime(2023, 1, 1, 0, 0, 0, 300000), datetime(2023, 1, 1, 0, 0, 0, 400000)]
    assert result == expected

def test_bulk_create_datetimes_with_combined_step():
    start = datetime(2023, 1, 1, 0, 0, 0)
    end = datetime(2023, 1, 2, 12, 0, 0)
    result = Datetime.bulk_create_datetimes(start, end, days=1, hours=12)
    expected = [datetime(2023, 1, 2, 12, 0, 0), datetime(2023, 1, 4, 0, 0, 0)]
    assert result == expected

def test_bulk_create_datetimes_single_step_exact_end():
    start = datetime(2023, 1, 1)
    end = datetime(2023, 1, 2)
    result = Datetime.bulk_create_datetimes(start, end, days=1)
    expected = [datetime(2023, 1, 2), datetime(2023, 1, 3)]
    assert result == expected


