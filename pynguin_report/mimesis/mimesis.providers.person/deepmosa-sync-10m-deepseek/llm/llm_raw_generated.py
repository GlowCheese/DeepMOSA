####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_surname_returns_string():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)

def test_surname_with_gender_male():
    person = Person()
    result = person.surname(Gender.MALE)
    assert isinstance(result, str)

def test_surname_with_gender_female():
    person = Person()
    result = person.surname(Gender.FEMALE)
    assert isinstance(result, str)

def test_surname_with_gender_none():
    person = Person()
    result = person.surname(None)
    assert isinstance(result, str)

def test_surname_different_calls_return_different_values():
    person = Person()
    result1 = person.surname()
    result2 = person.surname()
    assert result1 != result2

def test_surname_with_seed_returns_same_value():
    person = Person(seed=42)
    result1 = person.surname()
    person2 = Person(seed=42)
    result2 = person2.surname()
    assert result1 == result2

def test_surname_with_gender_and_seed():
    person = Person(seed=123)
    result1 = person.surname(Gender.MALE)
    person2 = Person(seed=123)
    result2 = person2.surname(Gender.MALE)
    assert result1 == result2

def test_surname_returns_valid_string_length():
    person = Person()
    result = person.surname()
    assert len(result) > 0

def test_surname_with_different_genders_return_strings():
    person = Person()
    male_result = person.surname(Gender.MALE)
    female_result = person.surname(Gender.FEMALE)
    assert isinstance(male_result, str)
    assert isinstance(female_result, str)

def test_surname_with_same_seed_and_gender_consistent():
    person1 = Person(seed=999)
    person2 = Person(seed=999)
    for _ in range(5):
        assert person1.surname(Gender.FEMALE) == person2.surname(Gender.FEMALE)

def test_surname_no_exception_with_valid_gender():
    person = Person()
    person.surname(Gender.MALE)
    person.surname(Gender.FEMALE)
    person.surname(None)


# LLM-generated content at query #2
#--------------------------

def test_email_default_domain():
    person = Person()
    email = person.email()
    assert "@" in email
    assert email.split("@")[1] in EMAIL_DOMAINS

def test_email_custom_domains():
    person = Person()
    custom_domains = ["@example.com", "@test.org"]
    email = person.email(domains=custom_domains)
    assert email.endswith(tuple(custom_domains))

def test_email_unique_without_seed():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False
    except ValueError as e:
        assert "You cannot use «unique» parameter with the seeded provider" in str(e)

def test_email_username_part():
    person = Person()
    email = person.email()
    username_part = email.split("@")[0]
    assert username_part.isalnum()

def test_email_domain_without_at_sign():
    person = Person()
    domains = ["example.com", "test.org"]
    email = person.email(domains=domains)
    assert email.startswith("@") == False
    assert any(email.endswith(domain) for domain in domains)


# LLM-generated content at query #3
#--------------------------

def test_patronymic_with_gender_male():
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert result is None or isinstance(result, str)

def test_patronymic_with_gender_female():
    person = Person()
    result = person.patronymic(Gender.FEMALE)
    assert result is None or isinstance(result, str)

def test_patronymic_without_gender():
    person = Person()
    result = person.patronymic()
    assert result is None or isinstance(result, str)

def test_patronymic_with_none_gender():
    person = Person()
    result = person.patronymic(None)
    assert result is None or isinstance(result, str)

def test_patronymic_returns_string_or_none():
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert result is None or isinstance(result, str)

def test_patronymic_with_invalid_gender():
    person = Person()
    try:
        person.patronymic("invalid")
    except Exception as e:
        assert isinstance(e, Exception)

def test_patronymic_seeded_randomness():
    person = Person(seed=42)
    result1 = person.patronymic(Gender.MALE)
    person2 = Person(seed=42)
    result2 = person2.patronymic(Gender.MALE)
    assert result1 == result2

def test_patronymic_different_genders_different_results():
    person = Person(seed=123)
    male_result = person.patronymic(Gender.MALE)
    female_result = person.patronymic(Gender.FEMALE)
    assert male_result != female_result or (male_result is None and female_result is None)


# LLM-generated content at query #4
#--------------------------

def test_nationality_returns_string():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)

def test_nationality_with_gender_male():
    person = Person()
    result = person.nationality(gender=Gender.MALE)
    assert isinstance(result, str)

def test_nationality_with_gender_female():
    person = Person()
    result = person.nationality(gender=Gender.FEMALE)
    assert isinstance(result, str)

def test_nationality_with_gender_none():
    person = Person()
    result = person.nationality(gender=None)
    assert isinstance(result, str)

def test_nationality_with_gender_enum():
    person = Person()
    result = person.nationality(gender=Gender.FEMALE)
    assert isinstance(result, str)

def test_nationality_different_calls_return_varied():
    person = Person()
    results = [person.nationality() for _ in range(10)]
    assert any(results[0] != other for other in results[1:])

def test_nationality_with_seed_returns_same():
    person = Person(seed=42)
    result1 = person.nationality()
    person2 = Person(seed=42)
    result2 = person2.nationality()
    assert result1 == result2


# LLM-generated content at query #5
#--------------------------

def test_email_with_default_domain():
    person = Person()
    email = person.email()
    assert "@" in email
    assert email.split("@")[1] in EMAIL_DOMAINS

def test_email_with_custom_domains():
    person = Person()
    custom_domains = ["@test.com", "@example.org"]
    email = person.email(domains=custom_domains)
    assert email.endswith(tuple(custom_domains))

def test_email_unique_without_seed():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=12345)
    try:
        person.email(unique=True)
        assert False
    except ValueError:
        assert True

def test_email_without_unique_and_seed():
    person = Person(seed=67890)
    email1 = person.email()
    person2 = Person(seed=67890)
    email2 = person2.email()
    assert email1 == email2

def test_email_domain_without_at_symbol():
    person = Person()
    domains = ["test.com", "example.org"]
    email = person.email(domains=domains)
    assert email.startswith(email.split("@")[0] + "@")
    assert email.split("@")[1] in domains

def test_email_username_part():
    person = Person()
    email = person.email()
    username_part = email.split("@")[0]
    assert username_part.isalnum()


# LLM-generated content at query #6
#--------------------------

def test_patronymic_returns_none_when_no_patronymics():
    person = Person()
    person._extract = lambda keys, default: []
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #7
#--------------------------

def test_username_default_mask():
    person = Person()
    result = person.username()
    assert isinstance(result, str)
    assert len(result) > 0

def test_username_custom_mask():
    person = Person()
    result = person.username(mask='C_C_d')
    parts = result.split('_')
    assert len(parts) == 3
    assert parts[0][0].isupper()
    assert parts[1][0].isupper()
    assert parts[2].isdigit()

def test_username_drange():
    person = Person()
    drange = (1900, 1950)
    result = person.username(mask='d', drange=drange)
    year = int(result)
    assert drange[0] <= year <= drange[1]

def test_username_invalid_drange_length():
    person = Person()
    try:
        person.username(drange=(1, 2, 3))
        assert False
    except ValueError as e:
        assert str(e) == "The drange parameter must contain only two integers."

def test_username_mask_without_required_tags():
    person = Person()
    try:
        person.username(mask='d.d')
        assert False
    except ValueError as e:
        assert str(e) == "Username mask must contain at least one of these: (C, U, l)."

def test_username_separators():
    person = Person()
    result = person.username(mask='C-U.l')
    assert any(c in result for c in '-.')

def test_username_lowercase_tag():
    person = Person()
    result = person.username(mask='l')
    assert result.islower()

def test_username_uppercase_tag():
    person = Person()
    result = person.username(mask='U')
    assert result.isupper()

def test_username_capitalized_tag():
    person = Person()
    result = person.username(mask='C')
    assert result[0].isupper() and result[1:].islower()


# LLM-generated content at query #8
#--------------------------

def test_email_default_domain():
    person = Person()
    email = person.email()
    assert "@" in email
    assert email.split("@")[1] in EMAIL_DOMAINS

def test_email_custom_domains():
    person = Person()
    custom_domains = ["example.com", "test.org"]
    email = person.email(domains=custom_domains)
    assert email.endswith("@example.com") or email.endswith("@test.org")

def test_email_unique_without_seed():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=12345)
    try:
        person.email(unique=True)
        assert False
    except ValueError as e:
        assert "You cannot use «unique» parameter with the seeded provider" in str(e)

def test_email_domain_without_at_symbol():
    person = Person()
    domains = ["example.com"]
    email = person.email(domains=domains)
    assert email.endswith("@example.com")

def test_email_domain_with_at_symbol():
    person = Person()
    domains = ["@example.com"]
    email = person.email(domains=domains)
    assert email.endswith("@example.com")

def test_email_username_format():
    person = Person()
    email = person.email()
    username_part = email.split("@")[0]
    assert any(char.isdigit() for char in username_part)
    assert any(char.isalpha() for char in username_part)

def test_email_no_domains_provided():
    person = Person()
    email = person.email(domains=[])
    assert email.split("@")[1] in EMAIL_DOMAINS


# LLM-generated content at query #9
#--------------------------

def test_username_default_mask():
    person = Person()
    result = person.username()
    assert isinstance(result, str)
    assert len(result) > 0

def test_username_custom_mask():
    person = Person()
    result = person.username(mask='C_C_d')
    parts = result.split('_')
    assert len(parts) == 3
    assert parts[0][0].isupper()
    assert parts[1][0].isupper()
    assert parts[2].isdigit()

def test_username_with_different_separators():
    person = Person()
    result = person.username(mask='U.l.d')
    parts = result.split('.')
    assert len(parts) == 3
    assert parts[0].isupper()
    assert parts[1].islower()
    assert parts[2].isdigit()

def test_username_drange_parameter():
    person = Person()
    drange = (1900, 1950)
    result = person.username(mask='l_l_d', drange=drange)
    parts = result.split('_')
    digit_part = int(parts[2])
    assert drange[0] <= digit_part <= drange[1]

def test_username_invalid_drange_length():
    person = Person()
    try:
        person.username(drange=(1, 2, 3))
        assert False
    except ValueError as e:
        assert str(e) == "The drange parameter must contain only two integers."

def test_username_mask_without_required_tags():
    person = Person()
    try:
        person.username(mask='###')
        assert False
    except ValueError as e:
        assert str(e) == "Username mask must contain at least one of these: (C, U, l)."

def test_username_unique_outputs():
    person = Person()
    results = set()
    for _ in range(100):
        results.add(person.username(mask='l_d'))
    assert len(results) > 1


# LLM-generated content at query #10
#--------------------------

def test_username_default_mask():
    person = Person()
    result = person.username()
    assert isinstance(result, str)
    assert len(result) > 0

def test_username_custom_mask_c_c_d():
    person = Person()
    result = person.username(mask='C_C_d')
    parts = result.split('_')
    assert len(parts) == 3
    assert parts[0][0].isupper()
    assert parts[1][0].isupper()
    assert parts[2].isdigit()

def test_username_custom_mask_u_l_d():
    person = Person()
    result = person.username(mask='U.l.d')
    parts = result.split('.')
    assert len(parts) == 3
    assert parts[0].isupper()
    assert parts[1].islower()
    assert parts[2].isdigit()

def test_username_custom_mask_l_l_d():
    person = Person()
    result = person.username(mask='l_l_d', drange=(1900, 2021))
    parts = result.split('_')
    assert len(parts) == 3
    assert parts[0].islower()
    assert parts[1].islower()
    assert parts[2].isdigit()
    year = int(parts[2])
    assert 1900 <= year <= 2021

def test_username_invalid_drange_length():
    person = Person()
    try:
        person.username(drange=(1, 2, 3))
        assert False
    except ValueError as e:
        assert str(e) == "The drange parameter must contain only two integers."

def test_username_mask_without_required_tags():
    person = Person()
    try:
        person.username(mask='###')
        assert False
    except ValueError as e:
        assert "Username mask must contain at least one of these: (C, U, l)." in str(e)

def test_username_mask_with_separators():
    person = Person()
    result = person.username(mask='C-U.l_d')
    assert any(c.isupper() for c in result)
    assert '-' in result or '.' in result or '_' in result

def test_username_drange_custom():
    person = Person()
    result = person.username(mask='d', drange=(5, 10))
    num = int(result)
    assert 5 <= num <= 10


# LLM-generated content at query #11
#--------------------------

def test_patronymic_returns_none_when_patronymics_list_is_empty():
    person = Person()
    person._extract = lambda keys, default: []
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #12
#--------------------------

def test_surname_without_gender():
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname


def test_surname_with_gender_male():
    person = Person()
    surname = person.surname(gender=Gender.MALE)
    assert isinstance(surname, str)
    assert surname


def test_surname_with_gender_female():
    person = Person()
    surname = person.surname(gender=Gender.FEMALE)
    assert isinstance(surname, str)
    assert surname


def test_surname_with_gender_none():
    person = Person()
    surname = person.surname(gender=None)
    assert isinstance(surname, str)
    assert surname


def test_surname_returns_string_from_surnames_list():
    person = Person()
    person._extract = lambda keys: ["Smith", "Johnson", "Williams"]
    surname = person.surname()
    assert surname in ["Smith", "Johnson", "Williams"]


def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person.validate_enum = lambda gender, enum_class: "male" if gender == Gender.MALE else "female"
    surname_male = person.surname(gender=Gender.MALE)
    assert surname_male in ["Smith", "Johnson"]
    surname_female = person.surname(gender=Gender.FEMALE)
    assert surname_female in ["Williams", "Brown"]


# LLM-generated content at query #13
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    mock_surnames = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: mock_surnames
    result = person.surname(gender=Gender.MALE)
    assert result in mock_surnames["male"]


# LLM-generated content at query #14
#--------------------------

def test_patronymic_with_male_gender():
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert result is None or isinstance(result, str)

def test_patronymic_with_female_gender():
    person = Person()
    result = person.patronymic(Gender.FEMALE)
    assert result is None or isinstance(result, str)

def test_patronymic_without_gender():
    person = Person()
    result = person.patronymic()
    assert result is None or isinstance(result, str)

def test_patronymic_with_none_gender():
    person = Person()
    result = person.patronymic(None)
    assert result is None or isinstance(result, str)

def test_patronymic_returns_string_when_available():
    person = Person(locale=Locale.RU)
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str)

def test_patronymic_returns_none_when_not_available():
    person = Person(locale=Locale.EN)
    result = person.patronymic(Gender.FEMALE)
    assert result is None


# LLM-generated content at query #15
#--------------------------

def test_nationality_returns_string():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)

def test_nationality_with_gender_male():
    person = Person()
    result = person.nationality(gender=Gender.MALE)
    assert isinstance(result, str)

def test_nationality_with_gender_female():
    person = Person()
    result = person.nationality(gender=Gender.FEMALE)
    assert isinstance(result, str)

def test_nationality_with_gender_none():
    person = Person()
    result = person.nationality(gender=None)
    assert isinstance(result, str)

def test_nationality_different_calls_return_different_values():
    person = Person()
    results = [person.nationality() for _ in range(10)]
    assert len(set(results)) > 1

def test_nationality_with_seed_returns_same_value():
    person = Person(seed=42)
    result1 = person.nationality()
    person2 = Person(seed=42)
    result2 = person2.nationality()
    assert result1 == result2


# LLM-generated content at query #16
#--------------------------

def test_surname_returns_string():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)

def test_surname_with_gender_male():
    person = Person()
    result = person.surname(gender=Gender.MALE)
    assert isinstance(result, str)

def test_surname_with_gender_female():
    person = Person()
    result = person.surname(gender=Gender.FEMALE)
    assert isinstance(result, str)

def test_surname_with_gender_none():
    person = Person()
    result = person.surname(gender=None)
    assert isinstance(result, str)

def test_surname_different_calls_return_different_values():
    person = Person()
    result1 = person.surname()
    result2 = person.surname()
    assert result1 != result2

def test_surname_with_seed_returns_same_value():
    person = Person(seed=42)
    result1 = person.surname()
    person2 = Person(seed=42)
    result2 = person2.surname()
    assert result1 == result2


# LLM-generated content at query #17
#--------------------------

def test_nationality_with_gender_separated_dict():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda value, enum_class: "male" if value is None else value.value
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality(gender=Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #18
#--------------------------

def test_nationality_with_gender_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["American"]}
    person.validate_enum = lambda gender, enum_class: "male"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality(gender=Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #19
#--------------------------

def test_nationality_with_dict_and_gender():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda value, enum_class: "male"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality(gender=Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #20
#--------------------------

def test_patronymic_returns_none_when_no_patronymics():
    person = Person()
    person._extract = lambda keys, default: []
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #21
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    mock_surnames = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: mock_surnames
    result = person.surname(gender=Gender.MALE)
    assert result in mock_surnames["male"]
    result = person.surname(gender=Gender.FEMALE)
    assert result in mock_surnames["female"]


# LLM-generated content at query #22
#--------------------------

def test_patronymic_with_male_gender():
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert result is None or isinstance(result, str)

def test_patronymic_with_female_gender():
    person = Person()
    result = person.patronymic(Gender.FEMALE)
    assert result is None or isinstance(result, str)

def test_patronymic_without_gender():
    person = Person()
    result = person.patronymic()
    assert result is None or isinstance(result, str)

def test_patronymic_with_none_gender():
    person = Person()
    result = person.patronymic(None)
    assert result is None or isinstance(result, str)

def test_patronymic_returns_string_when_available():
    person = Person(locale="ru")
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str)

def test_patronymic_returns_none_when_not_available():
    person = Person(locale="en")
    result = person.patronymic(Gender.MALE)
    assert result is None


# LLM-generated content at query #23
#--------------------------

def test_nationality_with_gender_separated_dict():
    person = Person()
    mock_nationalities = {"male": ["Russian", "American"], "female": ["French", "Italian"]}
    person._extract = lambda keys, default=None: mock_nationalities
    result = person.nationality(gender=Gender.MALE)
    assert result in mock_nationalities["male"]
    result = person.nationality(gender=Gender.FEMALE)
    assert result in mock_nationalities["female"]


# LLM-generated content at query #24
#--------------------------

def test_surname_with_dict_surnames():
    person = Person()
    surnames_dict = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: surnames_dict
    result = person.surname(gender=Gender.MALE)
    assert result in surnames_dict["male"]


# LLM-generated content at query #25
#--------------------------

def test_surname_without_gender():
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname


def test_surname_with_male_gender():
    person = Person()
    surname = person.surname(Gender.MALE)
    assert isinstance(surname, str)
    assert surname


def test_surname_with_female_gender():
    person = Person()
    surname = person.surname(Gender.FEMALE)
    assert isinstance(surname, str)
    assert surname


def test_surname_with_none_gender():
    person = Person()
    surname = person.surname(None)
    assert isinstance(surname, str)
    assert surname


def test_surname_returns_string_from_surnames_list():
    person = Person()
    surnames = person._extract(["surnames"])
    if isinstance(surnames, dict):
        surnames = surnames[Gender.MALE]
    surname = person.surname(Gender.MALE)
    assert surname in surnames


def test_surname_uses_random_choice():
    person = Person()
    person.random = Random(42)
    surname1 = person.surname()
    person.random = Random(42)
    surname2 = person.surname()
    assert surname1 == surname2


def test_surname_with_gender_separated_surnames():
    person = Person()
    extracted = person._extract(["surnames"])
    if isinstance(extracted, dict):
        male_surnames = extracted[Gender.MALE]
        female_surnames = extracted[Gender.FEMALE]
        male_surname = person.surname(Gender.MALE)
        female_surname = person.surname(Gender.FEMALE)
        assert male_surname in male_surnames
        assert female_surname in female_surnames


def test_surname_with_non_dict_surnames():
    person = Person()
    extracted = person._extract(["surnames"])
    if not isinstance(extracted, dict):
        surname = person.surname()
        assert surname in extracted
        surname_with_gender = person.surname(Gender.MALE)
        assert surname_with_gender in extracted


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_username_default_mask():
    person = Person()
    result = person.username()
    assert isinstance(result, str)
    assert len(result) > 0
    parts = result.split('_')
    assert len(parts) == 2
    assert parts[0].islower()
    assert parts[1].isdigit()
    assert 1800 <= int(parts[1]) <= 2100

def test_username_custom_mask():
    person = Person()
    result = person.username(mask='C_C_d')
    assert isinstance(result, str)
    parts = result.split('_')
    assert len(parts) == 3
    assert parts[0][0].isupper()
    assert parts[1][0].isupper()
    assert parts[2].isdigit()
    assert 1800 <= int(parts[2]) <= 2100

def test_username_with_dots():
    person = Person()
    result = person.username(mask='U.l.d')
    assert isinstance(result, str)
    parts = result.split('.')
    assert len(parts) == 3
    assert parts[0].isupper()
    assert parts[1].islower()
    assert parts[2].isdigit()
    assert 1800 <= int(parts[2]) <= 2100

def test_username_with_hyphen():
    person = Person()
    result = person.username(mask='l-l-d')
    assert isinstance(result, str)
    parts = result.split('-')
    assert len(parts) == 3
    assert parts[0].islower()
    assert parts[1].islower()
    assert parts[2].isdigit()
    assert 1800 <= int(parts[2]) <= 2100

def test_username_custom_drange():
    person = Person()
    result = person.username(mask='l_d', drange=(1900, 2021))
    assert isinstance(result, str)
    parts = result.split('_')
    assert len(parts) == 2
    assert parts[0].islower()
    assert parts[1].isdigit()
    assert 1900 <= int(parts[1]) <= 2021

def test_username_invalid_drange_length():
    person = Person()
    try:
        person.username(drange=(1800, 2100, 2200))
        assert False
    except ValueError as e:
        assert str(e) == "The drange parameter must contain only two integers."

def test_username_mask_without_required_tags():
    person = Person()
    try:
        person.username(mask='d.d.d')
        assert False
    except ValueError as e:
        assert str(e) == "Username mask must contain at least one of these: (C, U, l)."

def test_username_empty_mask():
    person = Person()
    result = person.username(mask='')
    assert isinstance(result, str)
    parts = result.split('_')
    assert len(parts) == 2
    assert parts[0].islower()
    assert parts[1].isdigit()
    assert 1800 <= int(parts[1]) <= 2100

def test_username_complex_mask():
    person = Person()
    result = person.username(mask='C_l-U.d')
    assert isinstance(result, str)
    assert '.' in result
    assert '-' in result
    assert '_' in result

def test_username_seeded_reproducible():
    person = Person(seed=42)
    result1 = person.username(mask='C_l_d')
    person2 = Person(seed=42)
    result2 = person2.username(mask='C_l_d')
    assert result1 == result2


# LLM-generated content at query #2
#--------------------------

def test_username_default_mask():
    person = Person()
    result = person.username()
    assert isinstance(result, str)
    assert len(result) > 0

def test_username_custom_mask():
    person = Person()
    result = person.username(mask='C_C_d')
    parts = result.split('_')
    assert len(parts) == 3
    assert parts[0][0].isupper()
    assert parts[1][0].isupper()
    assert parts[2].isdigit()

def test_username_with_dot_separator():
    person = Person()
    result = person.username(mask='U.l.d')
    parts = result.split('.')
    assert len(parts) == 3
    assert parts[0].isupper()
    assert parts[1].islower()
    assert parts[2].isdigit()

def test_username_with_hyphen_separator():
    person = Person()
    result = person.username(mask='l-l-d')
    parts = result.split('-')
    assert len(parts) == 3
    assert parts[0].islower()
    assert parts[1].islower()
    assert parts[2].isdigit()

def test_username_custom_drange():
    person = Person()
    drange = (1900, 1950)
    result = person.username(mask='l_d', drange=drange)
    parts = result.split('_')
    year = int(parts[1])
    assert drange[0] <= year <= drange[1]

def test_username_invalid_drange_length():
    person = Person()
    try:
        person.username(drange=(1, 2, 3))
        assert False
    except ValueError as e:
        assert str(e) == "The drange parameter must contain only two integers."

def test_username_mask_without_required_tags():
    person = Person()
    try:
        person.username(mask='d.d.d')
        assert False
    except ValueError as e:
        assert str(e) == "Username mask must contain at least one of these: (C, U, l)."

def test_username_empty_mask():
    person = Person()
    result = person.username(mask='')
    assert isinstance(result, str)
    assert len(result) > 0

def test_username_complex_mask():
    person = Person()
    result = person.username(mask='C_l-U.d')
    assert isinstance(result, str)
    assert len(result) > 0

def test_username_seeded_reproducible():
    person1 = Person(seed=42)
    person2 = Person(seed=42)
    result1 = person1.username(mask='C_d')
    result2 = person2.username(mask='C_d')
    assert result1 == result2


# LLM-generated content at query #3
#--------------------------

def test_patronymic_returns_string_or_none():
    person = Person()
    result = person.patronymic()
    assert result is None or isinstance(result, str)

def test_patronymic_with_gender_male():
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert result is None or isinstance(result, str)

def test_patronymic_with_gender_female():
    person = Person()
    result = person.patronymic(Gender.FEMALE)
    assert result is None or isinstance(result, str)

def test_patronymic_with_gender_none():
    person = Person()
    result = person.patronymic(None)
    assert result is None or isinstance(result, str)

def test_patronymic_returns_valid_patronymic():
    person = Person(locale=Locale.RU)
    result = person.patronymic(Gender.MALE)
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

def test_patronymic_for_locale_without_patronymics():
    person = Person(locale=Locale.EN)
    result = person.patronymic()
    assert result is None

def test_patronymic_uses_random_choice():
    person = Person(locale=Locale.RU)
    patronymics = ["Ivanovich", "Petrovich", "Sidorovich"]
    person.random.choice = lambda x: patronymics[0]
    result = person.patronymic(Gender.MALE)
    assert result == patronymics[0]


# LLM-generated content at query #4
#--------------------------

def test_email_with_default_domain():
    person = Person()
    email = person.email()
    assert "@" in email
    assert email.split("@")[1] in EMAIL_DOMAINS

def test_email_with_custom_domains():
    person = Person()
    custom_domains = ["example.com", "test.org"]
    email = person.email(domains=custom_domains)
    assert email.endswith(("@example.com", "@test.org"))

def test_email_unique_without_seed():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=12345)
    try:
        person.email(unique=True)
        assert False
    except ValueError as e:
        assert "You cannot use «unique» parameter with the seeded provider" in str(e)

def test_email_contains_username_and_domain():
    person = Person()
    email = person.email()
    parts = email.split("@")
    assert len(parts) == 2
    assert parts[0]
    assert parts[1]

def test_email_domain_starts_with_at():
    person = Person()
    domains = ["@example.com"]
    email = person.email(domains=domains)
    assert email.endswith("@example.com")

def test_email_domain_does_not_start_with_at():
    person = Person()
    domains = ["example.com"]
    email = person.email(domains=domains)
    assert email.endswith("@example.com")

def test_email_unique_uses_uuid():
    person = Person()
    email = person.email(unique=True)
    username_part = email.split("@")[0]
    assert len(username_part) == 32
    try:
        uuid.UUID(username_part)
        assert True
    except ValueError:
        assert False

def test_email_non_unique_uses_username():
    person = Person()
    email = person.email(unique=False)
    username_part = email.split("@")[0]
    assert any(c.isalpha() for c in username_part)
    assert any(c.isdigit() for c in username_part)


# LLM-generated content at query #5
#--------------------------

def test_person_constructor_without_arguments():
    person = Person()
    assert person is not None

def test_person_constructor_with_locale():
    person = Person(locale="en")
    assert person is not None

def test_person_constructor_with_seed():
    person = Person(seed=12345)
    assert person is not None

def test_person_constructor_with_locale_and_seed():
    person = Person(locale="en", seed=12345)
    assert person is not None

def test_person_constructor_with_arbitrary_keyword_arguments():
    person = Person(custom_arg="value")
    assert person is not None

def test_person_constructor_with_arbitrary_positional_arguments():
    person = Person("arg1", "arg2")
    assert person is not None

def test_person_constructor_meta_attributes():
    person = Person()
    assert person.Meta.name == "person"
    assert person.Meta.datafile == "person.json"


# LLM-generated content at query #6
#--------------------------

def test_username_with_uppercase_tag():
    person = Person()
    result = person.username(mask="U")
    assert result.isupper()


# LLM-generated content at query #7
#--------------------------

def test_nationality_without_gender():
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality


def test_nationality_with_male_gender():
    person = Person()
    nationality = person.nationality(Gender.MALE)
    assert isinstance(nationality, str)
    assert nationality


def test_nationality_with_female_gender():
    person = Person()
    nationality = person.nationality(Gender.FEMALE)
    assert isinstance(nationality, str)
    assert nationality


def test_nationality_with_other_gender():
    person = Person()
    nationality = person.nationality(Gender.OTHER)
    assert isinstance(nationality, str)
    assert nationality


def test_nationality_with_none_gender():
    person = Person()
    nationality = person.nationality(None)
    assert isinstance(nationality, str)
    assert nationality


def test_nationality_returns_different_values():
    person = Person()
    nationalities = {person.nationality() for _ in range(10)}
    assert len(nationalities) > 1


# LLM-generated content at query #8
#--------------------------

def test_surname_without_gender():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)
    assert result


def test_surname_with_gender_male():
    person = Person()
    result = person.surname(gender=Gender.MALE)
    assert isinstance(result, str)
    assert result


def test_surname_with_gender_female():
    person = Person()
    result = person.surname(gender=Gender.FEMALE)
    assert isinstance(result, str)
    assert result


def test_surname_with_gender_none():
    person = Person()
    result = person.surname(gender=None)
    assert isinstance(result, str)
    assert result


def test_surname_returns_string_from_surnames_list():
    person = Person()
    person._extract = lambda keys: ["Smith", "Johnson", "Williams"]
    person.random.choice = lambda lst: lst[0]
    result = person.surname()
    assert result == "Smith"


def test_surname_with_dict_surnames_and_gender_male():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person.random.choice = lambda lst: lst[0]
    person.validate_enum = lambda gender, enum_class: "male"
    result = person.surname(gender=Gender.MALE)
    assert result == "Smith"


def test_surname_with_dict_surnames_and_gender_female():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person.random.choice = lambda lst: lst[1]
    person.validate_enum = lambda gender, enum_class: "female"
    result = person.surname(gender=Gender.FEMALE)
    assert result == "Brown"


def test_surname_calls_random_choice():
    person = Person()
    person._extract = lambda keys: ["Smith", "Johnson", "Williams"]
    choice_called = False
    def mock_choice(lst):
        nonlocal choice_called
        choice_called = True
        return lst[0]
    person.random.choice = mock_choice
    person.surname()
    assert choice_called


# LLM-generated content at query #9
#--------------------------

def test_person_constructor_without_arguments():
    person = Person()
    assert person is not None

def test_person_constructor_with_locale():
    person = Person(locale="en")
    assert person is not None

def test_person_constructor_with_seed():
    person = Person(seed=12345)
    assert person is not None

def test_person_constructor_with_locale_and_seed():
    person = Person(locale="en", seed=12345)
    assert person is not None

def test_person_constructor_with_arbitrary_keyword_arguments():
    person = Person(custom_arg="value")
    assert person is not None

def test_person_constructor_with_arbitrary_positional_arguments():
    person = Person("arg1", "arg2")
    assert person is not None

def test_person_constructor_meta_attribute():
    person = Person()
    assert person.Meta.name == "person"
    assert person.Meta.datafile == "person.json"


# LLM-generated content at query #10
#--------------------------

def test_patronymic_with_gender_male():
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert result is None or isinstance(result, str)

def test_patronymic_with_gender_female():
    person = Person()
    result = person.patronymic(Gender.FEMALE)
    assert result is None or isinstance(result, str)

def test_patronymic_without_gender():
    person = Person()
    result = person.patronymic()
    assert result is None or isinstance(result, str)

def test_patronymic_with_invalid_gender():
    person = Person()
    try:
        person.patronymic("invalid")
    except Exception as e:
        assert isinstance(e, Exception)

def test_patronymic_returns_string_or_none():
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert result is None or isinstance(result, str)

def test_patronymic_with_locale_ru():
    person = Person(locale=Locale.RU)
    result = person.patronymic(Gender.MALE)
    assert result is not None and isinstance(result, str)

def test_patronymic_with_locale_uk():
    person = Person(locale=Locale.UK)
    result = person.patronymic(Gender.FEMALE)
    assert result is not None and isinstance(result, str)

def test_patronymic_with_non_supported_locale():
    person = Person(locale=Locale.EN)
    result = person.patronymic(Gender.MALE)
    assert result is None


# LLM-generated content at query #11
#--------------------------

def test_surname_with_dict_surnames():
    mock_extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    mock_validate_enum = lambda gender, enum: "male" if gender is None else gender.value
    mock_random_choice = lambda seq: seq[0]
    person = Person()
    person._extract = mock_extract
    person.validate_enum = mock_validate_enum
    person.random.choice = mock_random_choice
    result = person.surname()
    assert result == "Smith"


# LLM-generated content at query #12
#--------------------------

def test_nationality_returns_string():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)

def test_nationality_with_gender_male():
    person = Person()
    result = person.nationality(gender=Gender.MALE)
    assert isinstance(result, str)

def test_nationality_with_gender_female():
    person = Person()
    result = person.nationality(gender=Gender.FEMALE)
    assert isinstance(result, str)

def test_nationality_with_gender_none():
    person = Person()
    result = person.nationality(gender=None)
    assert isinstance(result, str)

def test_nationality_with_gender_enum():
    person = Person()
    result = person.nationality(gender=Gender.MALE)
    assert isinstance(result, str)
    result = person.nationality(gender=Gender.FEMALE)
    assert isinstance(result, str)

def test_nationality_different_calls_return_different_values():
    person = Person()
    result1 = person.nationality()
    result2 = person.nationality()
    assert result1 != result2

def test_nationality_with_seed_returns_same_value():
    person = Person(seed=42)
    result1 = person.nationality()
    person2 = Person(seed=42)
    result2 = person2.nationality()
    assert result1 == result2

def test_nationality_with_gender_and_seed():
    person = Person(seed=123)
    result1 = person.nationality(gender=Gender.FEMALE)
    person2 = Person(seed=123)
    result2 = person2.nationality(gender=Gender.FEMALE)
    assert result1 == result2


# LLM-generated content at query #13
#--------------------------

def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=12345)
    exception_raised = False
    try:
        person.email(unique=True)
    except ValueError as e:
        exception_raised = True
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"
    assert exception_raised


# LLM-generated content at query #14
#--------------------------

def test_username_default_mask():
    person = Person()
    result = person.username()
    assert isinstance(result, str)
    assert len(result) > 0
    assert "_" in result
    parts = result.split("_")
    assert len(parts) == 2
    assert parts[0].islower()
    assert parts[1].isdigit()
    assert 1800 <= int(parts[1]) <= 2100

def test_username_custom_mask():
    person = Person()
    result = person.username(mask="C_C_d")
    assert isinstance(result, str)
    assert len(result) > 0
    parts = result.split("_")
    assert len(parts) == 3
    assert parts[0][0].isupper()
    assert parts[1][0].isupper()
    assert parts[2].isdigit()

def test_username_custom_drange():
    person = Person()
    result = person.username(drange=(1900, 2000))
    assert isinstance(result, str)
    parts = result.split("_")
    digit_part = parts[1]
    assert 1900 <= int(digit_part) <= 2000

def test_username_invalid_drange_length():
    person = Person()
    try:
        person.username(drange=(1800, 2100, 2200))
        assert False
    except ValueError as e:
        assert str(e) == "The drange parameter must contain only two integers."

def test_username_mask_without_required_tags():
    person = Person()
    try:
        person.username(mask="d.d.d")
        assert False
    except ValueError as e:
        assert str(e) == "Username mask must contain at least one of these: (C, U, l)."

def test_username_mask_with_separators():
    person = Person()
    result = person.username(mask="U.l.d")
    assert isinstance(result, str)
    assert "." in result
    parts = result.split(".")
    assert len(parts) == 3
    assert parts[0].isupper()
    assert parts[1].islower()
    assert parts[2].isdigit()

def test_username_mask_lowercase_only():
    person = Person()
    result = person.username(mask="l_l_l")
    assert isinstance(result, str)
    parts = result.split("_")
    assert len(parts) == 3
    assert all(part.islower() for part in parts)

def test_username_mask_uppercase_only():
    person = Person()
    result = person.username(mask="U_U_U")
    assert isinstance(result, str)
    parts = result.split("_")
    assert len(parts) == 3
    assert all(part.isupper() for part in parts)

def test_username_mask_capitalized_only():
    person = Person()
    result = person.username(mask="C_C_C")
    assert isinstance(result, str)
    parts = result.split("_")
    assert len(parts) == 3
    assert all(part[0].isupper() for part in parts)

def test_username_mask_mixed():
    person = Person()
    result = person.username(mask="C_l-U.d")
    assert isinstance(result, str)
    assert "-" in result
    assert "." in result
    assert "_" in result


# LLM-generated content at query #15
#--------------------------

def test_patronymic_returns_none_when_patronymics_is_empty():
    person = Person()
    person._extract = lambda keys, default: []
    result = person.patronymic()
    assert result is None


# LLM-generated content at query #16
#--------------------------

def test_init_calls_super_init():
    from unittest.mock import Mock
    mock_args = (1, 2, 3)
    mock_kwargs = {'locale': 'en_US', 'seed': 42}
    mock_super = Mock()
    original_super = __builtins__['super']
    __builtins__['super'] = Mock(return_value=mock_super)
    person = Person(*mock_args, **mock_kwargs)
    mock_super().__init__.assert_called_once_with(*mock_args, **mock_kwargs)
    __builtins__['super'] = original_super


# LLM-generated content at query #17
#--------------------------

def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=12345)
    exception_raised = False
    try:
        person.email(unique=True)
    except ValueError as e:
        exception_raised = True
        error_message = str(e)
    assert exception_raised == True
    assert error_message == "You cannot use «unique» parameter with the seeded provider"


# LLM-generated content at query #18
#--------------------------

def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=12345)
    exception_raised = False
    try:
        person.email(unique=True)
    except ValueError as e:
        exception_raised = True
        error_message = str(e)
    assert exception_raised
    assert error_message == "You cannot use «unique» parameter with the seeded provider"


# LLM-generated content at query #19
#--------------------------

def test_patronymic_returns_none_when_patronymics_list_is_empty():
    mock_provider = Person()
    mock_provider.validate_enum = lambda gender, enum: Gender.MALE
    mock_provider._extract = lambda keys, default: []
    mock_provider.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = mock_provider.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #20
#--------------------------

def test_nationality_without_gender():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)
    assert len(result) > 0

def test_nationality_with_male_gender():
    person = Person()
    result = person.nationality(gender=Gender.MALE)
    assert isinstance(result, str)
    assert len(result) > 0

def test_nationality_with_female_gender():
    person = Person()
    result = person.nationality(gender=Gender.FEMALE)
    assert isinstance(result, str)
    assert len(result) > 0

def test_nationality_with_none_gender():
    person = Person()
    result = person.nationality(gender=None)
    assert isinstance(result, str)
    assert len(result) > 0

def test_nationality_returns_different_values():
    person = Person()
    results = set()
    for _ in range(10):
        results.add(person.nationality())
    assert len(results) > 1


# LLM-generated content at query #21
#--------------------------

def test___init__():
    person = Person()
    assert isinstance(person, Person)


# LLM-generated content at query #22
#--------------------------

def test_nationality_with_gender_separated_dict():
    person = Person()
    mock_nationalities = {"MALE": ["American", "British"], "FEMALE": ["American", "British"]}
    person._extract = lambda keys, default=None: mock_nationalities
    result = person.nationality(gender=Gender.MALE)
    assert result in mock_nationalities["MALE"]
    result = person.nationality(gender=Gender.FEMALE)
    assert result in mock_nationalities["FEMALE"]


# LLM-generated content at query #23
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person.validate_enum = lambda gender, enum_class: "male" if gender == Gender.MALE else "female"
    person.random = type("Random", (), {"choice": lambda lst: lst[0]})()
    result = person.surname(gender=Gender.MALE)
    assert result == "Smith"


# LLM-generated content at query #24
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person.validate_enum = lambda gender, enum_class: "male" if gender == Gender.MALE else "female"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.surname(gender=Gender.MALE)
    assert result == "Smith"


# LLM-generated content at query #25
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    mock_surnames = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: mock_surnames
    result = person.surname(gender=Gender.MALE)
    assert result in mock_surnames["male"]
    result = person.surname(gender=Gender.FEMALE)
    assert result in mock_surnames["female"]


