####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_locale_with_valid_string():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)
    assert str(result) == "en_US"

def test_validate_locale_with_valid_locale():
    locale = Locale("fr_FR")
    result = validate_locale(locale)
    assert result is locale

def test_validate_locale_with_invalid_string():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError as e:
        assert str(e) == "invalid_locale"

def test_validate_locale_with_invalid_type():
    try:
        validate_locale(123)
        assert False
    except LocaleError as e:
        assert str(e) == "123"


# LLM-generated content at query #2
#--------------------------

def test_validate_locale_with_valid_locale_object():
    locale_obj = Locale("en_US")
    result = validate_locale(locale_obj)
    assert result == locale_obj

def test_validate_locale_with_valid_locale_string():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)
    assert str(result) == "en_US"

def test_validate_locale_with_invalid_locale_string():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError as e:
        assert str(e) == "invalid_locale"

def test_validate_locale_with_non_string_non_locale():
    try:
        validate_locale(123)
        assert False
    except LocaleError as e:
        assert e.args[0] == 123


# LLM-generated content at query #3
#--------------------------

def test_validate_locale_with_valid_string_locale():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)

def test_validate_locale_with_locale_object():
    locale_obj = Locale("en_US")
    result = validate_locale(locale_obj)
    assert result is locale_obj

def test_validate_locale_with_invalid_string_locale():
    try:
        validate_locale("invalid_locale_string")
        assert False
    except LocaleError:
        assert True

def test_validate_locale_with_non_string_non_locale():
    try:
        validate_locale(123)
        assert False
    except LocaleError:
        assert True


# LLM-generated content at query #4
#--------------------------

def test_validate_locale_with_valid_string_locale():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)

def test_validate_locale_with_locale_object():
    locale_obj = Locale("en_US")
    result = validate_locale(locale_obj)
    assert result is locale_obj

def test_validate_locale_with_invalid_string_locale():
    try:
        validate_locale("invalid_locale")
    except LocaleError:
        pass
    else:
        assert False

def test_validate_locale_with_non_string_non_locale():
    try:
        validate_locale(123)
    except LocaleError:
        pass
    else:
        assert False


# LLM-generated content at query #5
#--------------------------

def test_validate_locale_with_valid_string_locale():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)

def test_validate_locale_with_locale_object():
    locale_obj = Locale("en_US")
    result = validate_locale(locale_obj)
    assert result == locale_obj

def test_validate_locale_with_invalid_string_locale():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError:
        pass

def test_validate_locale_with_non_string_non_locale():
    try:
        validate_locale(123)
        assert False
    except LocaleError:
        pass


# LLM-generated content at query #6
#--------------------------

def test_validate_locale_returns_locale_when_locale_is_locale_instance():
    locale = Locale("en_US")
    result = validate_locale(locale)
    assert result == locale

def test_validate_locale_returns_locale_when_locale_is_valid_string():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)

def test_validate_locale_raises_locale_error_when_locale_is_invalid_string():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError:
        assert True

def test_validate_locale_raises_locale_error_when_locale_is_neither_string_nor_locale():
    try:
        validate_locale(123)
        assert False
    except LocaleError:
        assert True


# LLM-generated content at query #7
#--------------------------

def test_validate_locale_with_valid_string_locale():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)

def test_validate_locale_with_locale_object():
    locale_obj = Locale("en_US")
    result = validate_locale(locale_obj)
    assert result is locale_obj

def test_validate_locale_with_invalid_string_locale():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError:
        assert True

def test_validate_locale_with_non_string_non_locale():
    try:
        validate_locale(123)
        assert False
    except LocaleError:
        assert True


# LLM-generated content at query #8
#--------------------------

def test_validate_locale_with_valid_locale_object():
    locale_obj = Locale("en_US")
    result = validate_locale(locale_obj)
    assert result == locale_obj

def test_validate_locale_with_valid_locale_string():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)
    assert str(result) == "en_US"

def test_validate_locale_with_invalid_locale_string():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError:
        assert True

def test_validate_locale_with_non_string_non_locale():
    try:
        validate_locale(123)
        assert False
    except LocaleError:
        assert True


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_validate_locale_with_valid_string():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)
    assert str(result) == "en_US"

def test_validate_locale_with_valid_locale():
    locale = Locale("fr_FR")
    result = validate_locale(locale)
    assert result is locale

def test_validate_locale_with_invalid_string():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError as e:
        assert str(e) == "invalid_locale"

def test_validate_locale_with_invalid_type():
    try:
        validate_locale(123)
        assert False
    except LocaleError as e:
        assert str(e) == "123"


# LLM-generated content at query #2
#--------------------------

def test_validate_locale_with_valid_string():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)
    assert str(result) == "en_US"

def test_validate_locale_with_valid_locale():
    locale = Locale("fr_FR")
    result = validate_locale(locale)
    assert result is locale

def test_validate_locale_with_invalid_string():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError as e:
        assert str(e.locale) == "invalid_locale"

def test_validate_locale_with_invalid_type():
    try:
        validate_locale(123)
        assert False
    except LocaleError as e:
        assert e.locale == 123


# LLM-generated content at query #3
#--------------------------

def test_validate_locale_with_valid_string():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)
    assert str(result) == "en_US"

def test_validate_locale_with_valid_locale():
    locale = Locale("en_US")
    result = validate_locale(locale)
    assert result is locale

def test_validate_locale_with_invalid_string():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError as e:
        assert str(e) == "invalid_locale"

def test_validate_locale_with_invalid_type():
    try:
        validate_locale(123)
        assert False
    except LocaleError as e:
        assert str(e) == "123"


# LLM-generated content at query #4
#--------------------------

def test_validate_locale_with_valid_locale_object():
    locale = Locale("en_US")
    result = validate_locale(locale)
    assert result == locale

def test_validate_locale_with_valid_locale_string():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)
    assert str(result) == "en_US"

def test_validate_locale_with_invalid_locale_string():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError:
        assert True

def test_validate_locale_with_non_string_non_locale():
    try:
        validate_locale(123)
        assert False
    except LocaleError:
        assert True


# LLM-generated content at query #5
#--------------------------

def test_validate_locale_with_valid_locale_object():
    locale_obj = Locale("en_US")
    result = validate_locale(locale_obj)
    assert result == locale_obj

def test_validate_locale_with_valid_locale_string():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)
    assert str(result) == "en_US"

def test_validate_locale_with_invalid_locale_string():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError as e:
        assert str(e) == "invalid_locale"

def test_validate_locale_with_invalid_type():
    try:
        validate_locale(123)
        assert False
    except LocaleError as e:
        assert e.args[0] == 123


# LLM-generated content at query #6
#--------------------------

def test_validate_locale_with_valid_locale_object():
    locale = Locale("en_US")
    result = validate_locale(locale)
    assert result == locale

def test_validate_locale_with_valid_locale_string():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)
    assert str(result) == "en_US"

def test_validate_locale_with_invalid_locale_string():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError:
        assert True

def test_validate_locale_with_non_string_non_locale():
    try:
        validate_locale(123)
        assert False
    except LocaleError:
        assert True

def test_validate_locale_with_none():
    try:
        validate_locale(None)
        assert False
    except LocaleError:
        assert True


# LLM-generated content at query #7
#--------------------------

def test_validate_locale_returns_locale_when_locale_is_locale_object():
    locale_object = Locale("en_US")
    result = validate_locale(locale_object)
    assert result == locale_object


# LLM-generated content at query #8
#--------------------------

def test_validate_locale_with_valid_locale_object():
    locale_obj = Locale("en_US")
    result = validate_locale(locale_obj)
    assert result == locale_obj

def test_validate_locale_with_valid_locale_string():
    result = validate_locale("en_US")
    assert isinstance(result, Locale)
    assert str(result) == "en_US"

def test_validate_locale_with_invalid_locale_string():
    try:
        validate_locale("invalid_locale")
        assert False
    except LocaleError as e:
        assert str(e) == "invalid_locale"

def test_validate_locale_with_non_string_non_locale():
    try:
        validate_locale(123)
        assert False
    except LocaleError as e:
        assert e.args[0] == 123


