####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_person_constructor():
    person = Person()
    assert isinstance(person, Person)
    assert isinstance(person, BaseDataProvider)

def test_person_constructor_with_locale():
    person = Person(locale="en")
    assert person.locale == "en"

def test_person_constructor_with_seed():
    person = Person(seed=42)
    assert person.seed == 42

def test_person_constructor_with_locale_and_seed():
    person = Person(locale="ru", seed=123)
    assert person.locale == "ru"
    assert person.seed == 123


# LLM-generated content at query #2
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

def test_email_without_at_in_domain():
    person = Person()
    domains = ["test.com", "example.org"]
    email = person.email(domains=domains)
    assert email.startswith("@") is False
    assert any(email.endswith(domain) for domain in domains)

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

def test_email_username_part():
    person = Person()
    email = person.email()
    username_part = email.split("@")[0]
    assert username_part.isalnum() or "_" in username_part or "-" in username_part or "." in username_part

def test_email_with_empty_domains_list():
    person = Person()
    email = person.email(domains=[])
    assert email.endswith(tuple(EMAIL_DOMAINS))

def test_email_with_single_domain():
    person = Person()
    single_domain = ["@single.test"]
    email = person.email(domains=single_domain)
    assert email.endswith("@single.test")


# LLM-generated content at query #3
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

def test_surname_uses_random_choice():
    person = Person()
    surnames = ["Smith", "Johnson", "Williams"]
    person._extract = lambda keys: surnames
    person.random.choice = lambda seq: seq[0]
    result = person.surname()
    assert result == "Smith"

def test_surname_with_dict_structure():
    person = Person()
    surnames_dict = {"male": ["Smith"], "female": ["Johnson"]}
    person._extract = lambda keys: surnames_dict
    person.random.choice = lambda seq: seq[0]
    result_male = person.surname(Gender.MALE)
    result_female = person.surname(Gender.FEMALE)
    assert result_male == "Smith"
    assert result_female == "Johnson"


# LLM-generated content at query #4
#--------------------------

def test_surname_with_dict_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person.validate_enum = lambda gender, enum: "male" if gender is None else gender.value
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.surname()
    assert result == "Smith"


# LLM-generated content at query #5
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

def test_patronymic_returns_string_when_available():
    person = Person(locale="ru")
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str)

def test_patronymic_returns_none_when_not_available():
    person = Person(locale="en")
    result = person.patronymic(Gender.FEMALE)
    assert result is None


# LLM-generated content at query #6
#--------------------------

def test_email_generates_valid_format():
    person = Person()
    email = person.email()
    assert "@" in email
    assert "." in email
    parts = email.split("@")
    assert len(parts) == 2
    assert parts[1] in EMAIL_DOMAINS or parts[1][1:] in EMAIL_DOMAINS

def test_email_with_custom_domains():
    person = Person()
    custom_domains = ["example.com", "test.org"]
    email = person.email(domains=custom_domains)
    domain_part = email.split("@")[1]
    assert domain_part in custom_domains

def test_email_unique_flag_without_seed():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_flag_with_seed_raises_value_error():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False
    except ValueError as e:
        assert "You cannot use «unique» parameter with the seeded provider" in str(e)

def test_email_without_unique_flag_and_seed():
    person = Person(seed=42)
    email1 = person.email()
    email2 = person.email()
    assert email1 == email2

def test_email_uses_username_pattern():
    person = Person()
    email = person.email()
    local_part = email.split("@")[0]
    assert any(c.isdigit() for c in local_part)
    assert any(c.isalpha() for c in local_part)

def test_email_domain_starts_with_at():
    person = Person()
    email = person.email(domains=["@example.com"])
    assert email.endswith("@example.com")

def test_email_domain_without_at():
    person = Person()
    email = person.email(domains=["example.com"])
    assert email.endswith("@example.com")


# LLM-generated content at query #7
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person.validate_enum = lambda gender, enum_class: "male" if gender == Gender.MALE else "female"
    person.random = type("Random", (), {"choice": lambda lst: lst[0]})()
    result = person.surname(gender=Gender.MALE)
    assert result == "Smith"


# LLM-generated content at query #8
#--------------------------

def test_nationality_without_gender():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)
    assert result in person._extract(["nationality"])

def test_nationality_with_male_gender():
    person = Person()
    result = person.nationality(Gender.MALE)
    nationalities = person._extract(["nationality"])
    if isinstance(nationalities, dict):
        assert result in nationalities[Gender.MALE]
    else:
        assert result in nationalities

def test_nationality_with_female_gender():
    person = Person()
    result = person.nationality(Gender.FEMALE)
    nationalities = person._extract(["nationality"])
    if isinstance(nationalities, dict):
        assert result in nationalities[Gender.FEMALE]
    else:
        assert result in nationalities

def test_nationality_with_none_gender():
    person = Person()
    result = person.nationality(None)
    nationalities = person._extract(["nationality"])
    if isinstance(nationalities, dict):
        assert result in nationalities[Gender.MALE] or result in nationalities[Gender.FEMALE]
    else:
        assert result in nationalities

def test_nationality_returns_string():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)

def test_nationality_randomness():
    person = Person()
    results = {person.nationality() for _ in range(10)}
    assert len(results) > 1


# LLM-generated content at query #9
#--------------------------

def test_patronymic_returns_none_when_patronymics_is_empty():
    person = Person()
    person._extract = lambda keys, default: []
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


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
    result = person.patronymic(Gender.FEMALE)
    assert result is None


# LLM-generated content at query #11
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

def test_surname_with_specific_locale():
    person = Person(locale=Locale.RU)
    result = person.surname()
    assert isinstance(result, str)

def test_surname_gender_specific_surnames_exist():
    person = Person(locale=Locale.RU)
    male_surname = person.surname(Gender.MALE)
    female_surname = person.surname(Gender.FEMALE)
    assert isinstance(male_surname, str)
    assert isinstance(female_surname, str)

def test_surname_last_name_alias():
    person = Person()
    surname_result = person.surname()
    last_name_result = person.last_name()
    assert isinstance(surname_result, str)
    assert isinstance(last_name_result, str)


# LLM-generated content at query #12
#--------------------------

def test_patronymic_returns_none_when_no_patronymics():
    person = Person()
    person._extract = lambda keys, default: []
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #13
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

def test_nationality_with_gender_returns_different_values():
    person = Person()
    male_results = set()
    female_results = set()
    for _ in range(10):
        male_results.add(person.nationality(gender=Gender.MALE))
        female_results.add(person.nationality(gender=Gender.FEMALE))
    assert len(male_results) > 1
    assert len(female_results) > 1


# LLM-generated content at query #14
#--------------------------

def test_nationality_with_dict():
    person = Person()
    mock_nationalities = {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person._extract = lambda keys, default=None: mock_nationalities
    result = person.nationality(gender=Gender.MALE)
    assert result in mock_nationalities["male"]
    result = person.nationality(gender=Gender.FEMALE)
    assert result in mock_nationalities["female"]


# LLM-generated content at query #15
#--------------------------

def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False
    except ValueError as e:
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"


# LLM-generated content at query #16
#--------------------------

def test_nationality_with_dict_and_gender():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["French", "Italian"]}
    person.validate_enum = lambda gender, enum_class: "male"
    person.random.choice = lambda items: items[0]
    result = person.nationality(gender=Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #17
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
    email = person.email(domains=["example.com"])
    assert email.endswith("@example.com")

def test_email_domain_with_at_symbol():
    person = Person()
    email = person.email(domains=["@example.com"])
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
    assert "@" in email
    assert email.split("@")[1] in EMAIL_DOMAINS

def test_email_unique_uses_uuid():
    person = Person()
    email = person.email(unique=True)
    username_part = email.split("@")[0]
    try:
        uuid.UUID(username_part, version=4)
        assert True
    except ValueError:
        assert False

def test_email_non_unique_uses_username():
    person = Person()
    email = person.email(unique=False)
    username_part = email.split("@")[0]
    assert len(username_part) > 0
    assert not (username_part.replace("_", "").replace(".", "").replace("-", "").isalnum() and any(c.isdigit() for c in username_part) and any(c.isalpha() for c in username_part))


# LLM-generated content at query #18
#--------------------------

def test_patronymic_returns_none_when_patronymics_list_is_empty():
    person = Person()
    person._extract = lambda keys, default: []
    person.validate_enum = lambda gender, enum: Gender.MALE
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #19
#--------------------------

def test_nationality_with_gender_dict():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda value, enum_class: "male" if value == Gender.MALE else "female"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality(Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #20
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

def test_surname_with_gender_enum():
    person = Person()
    result = person.surname(Gender.MALE)
    assert isinstance(result, str)

def test_surname_returns_different_values():
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

def test_surname_uses_extracted_data():
    person = Person()
    surnames = person._extract(["surnames"])
    result = person.surname()
    if isinstance(surnames, dict):
        assert result in surnames[Gender.MALE] or result in surnames[Gender.FEMALE]
    else:
        assert result in surnames

def test_surname_with_dict_surnames_and_gender():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    result = person.surname(Gender.MALE)
    assert result in ["Smith", "Johnson"]

def test_surname_with_list_surnames():
    person = Person()
    person._extract = lambda keys, default=None: ["Smith", "Johnson", "Williams"]
    result = person.surname()
    assert result in ["Smith", "Johnson", "Williams"]

def test_surname_with_dict_surnames_no_gender():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Smith"], "female": ["Johnson"]}
    result = person.surname()
    assert result in ["Smith", "Johnson"]


# LLM-generated content at query #21
#--------------------------

def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=42)
    exception_raised = False
    try:
        person.email(unique=True)
    except ValueError as e:
        exception_raised = True
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"
    assert exception_raised


# LLM-generated content at query #22
#--------------------------

def test_surname_with_dict_surnames_and_gender():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.surname(gender=Gender.MALE)
    assert result == "Smith"


# LLM-generated content at query #23
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
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False
    except ValueError:
        assert True

def test_email_without_unique_and_seed():
    person = Person(seed=42)
    email1 = person.email()
    person2 = Person(seed=42)
    email2 = person2.email()
    assert email1 == email2

def test_email_domain_without_at_symbol():
    person = Person()
    domains = ["test.com", "example.org"]
    email = person.email(domains=domains)
    assert email.startswith("@") == False
    assert any(email.endswith(domain) for domain in domains)

def test_email_domain_with_at_symbol():
    person = Person()
    domains = ["@test.com", "@example.org"]
    email = person.email(domains=domains)
    assert email.startswith("@") == False
    assert any(email.endswith(domain) for domain in domains)

def test_email_username_format():
    person = Person()
    email = person.email()
    username_part = email.split("@")[0]
    assert username_part.isalnum()


# LLM-generated content at query #24
#--------------------------

def test_patronymic_returns_none_when_patronymics_list_is_empty():
    person = Person()
    person._extract = lambda keys, default: []
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #25
#--------------------------

def test_nationality_without_gender():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)
    assert result


def test_nationality_with_male_gender():
    person = Person()
    result = person.nationality(gender=Gender.MALE)
    assert isinstance(result, str)
    assert result


def test_nationality_with_female_gender():
    person = Person()
    result = person.nationality(gender=Gender.FEMALE)
    assert isinstance(result, str)
    assert result


def test_nationality_with_other_gender():
    person = Person()
    result = person.nationality(gender=Gender.OTHER)
    assert isinstance(result, str)
    assert result


def test_nationality_with_none_gender():
    person = Person()
    result = person.nationality(gender=None)
    assert isinstance(result, str)
    assert result


def test_nationality_returns_different_values():
    person = Person()
    results = [person.nationality() for _ in range(10)]
    assert len(set(results)) > 1


def test_nationality_with_seed():
    person = Person(seed=42)
    result1 = person.nationality()
    person2 = Person(seed=42)
    result2 = person2.nationality()
    assert result1 == result2


# LLM-generated content at query #26
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

def test_patronymic_returns_string_when_available():
    person = Person(locale="ru")
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str)

def test_patronymic_returns_none_when_not_available():
    person = Person(locale="en")
    result = person.patronymic(Gender.FEMALE)
    assert result is None


# LLM-generated content at query #27
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    mock_surnames = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: mock_surnames
    result = person.surname(gender=Gender.MALE)
    assert result in mock_surnames["male"]


# LLM-generated content at query #28
#--------------------------

def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=42)
    exception_raised = False
    try:
        person.email(unique=True)
    except ValueError as e:
        exception_raised = True
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"
    assert exception_raised


# LLM-generated content at query #29
#--------------------------

def test_nationality_with_dict():
    person = Person()
    person._extract = lambda keys: {"MALE": ["Russian"], "FEMALE": ["American"]}
    person.validate_enum = lambda gender, enum_class: "MALE"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality(gender="MALE")
    assert result == "Russian"

def test_nationality_with_list():
    person = Person()
    person._extract = lambda keys: ["Russian", "American"]
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality()
    assert result == "Russian"

def test_nationality_with_dict_and_none_gender():
    person = Person()
    person._extract = lambda keys: {"MALE": ["Russian"], "FEMALE": ["American"]}
    person.validate_enum = lambda gender, enum_class: "MALE"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality(gender=None)
    assert result == "Russian"


# LLM-generated content at query #30
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    mock_surnames = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: mock_surnames
    result = person.surname(gender=Gender.MALE)
    assert result in mock_surnames["male"]
    result = person.surname(gender=Gender.FEMALE)
    assert result in mock_surnames["female"]


# LLM-generated content at query #31
#--------------------------

def test_nationality_with_gender_separated_dict():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Russian", "American"], "female": ["French", "Italian"]}
    person.validate_enum = lambda value, enum_class: "male" if value == Gender.MALE else "female"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality(Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #32
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

def test_email_with_custom_domains_without_at():
    person = Person()
    custom_domains = ["test.com", "example.org"]
    email = person.email(domains=custom_domains)
    assert any(email.endswith(domain) for domain in custom_domains)

def test_email_unique_without_seed():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_raises_error_with_seed():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False
    except ValueError as e:
        assert "You cannot use «unique» parameter with the seeded provider" in str(e)

def test_email_not_unique_with_seed():
    person = Person(seed=42)
    email1 = person.email(unique=False)
    email2 = person.email(unique=False)
    assert email1 == email2

def test_email_username_part():
    person = Person()
    email = person.email()
    username_part = email.split("@")[0]
    assert username_part.isalnum()


# LLM-generated content at query #33
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
    person._extract = lambda keys, default=None: patronymics
    person.random.choice = lambda seq: seq[0]
    result = person.patronymic(Gender.MALE)
    assert result == "Ivanovich"


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
    assert 1800 <= int(parts[2]) <= 2100

def test_username_mask_with_dots():
    person = Person()
    result = person.username(mask="U.l.d")
    assert isinstance(result, str)
    assert len(result) > 0
    parts = result.split(".")
    assert len(parts) == 3
    assert parts[0].isupper()
    assert parts[1].islower()
    assert parts[2].isdigit()
    assert 1800 <= int(parts[2]) <= 2100

def test_username_mask_with_hyphen():
    person = Person()
    result = person.username(mask="l-l-d")
    assert isinstance(result, str)
    assert len(result) > 0
    parts = result.split("-")
    assert len(parts) == 3
    assert parts[0].islower()
    assert parts[1].islower()
    assert parts[2].isdigit()
    assert 1800 <= int(parts[2]) <= 2100

def test_username_custom_drange():
    person = Person()
    result = person.username(mask="l_d", drange=(1900, 2021))
    assert isinstance(result, str)
    assert len(result) > 0
    parts = result.split("_")
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
        person.username(mask="d.d.d")
        assert False
    except ValueError as e:
        assert str(e) == "Username mask must contain at least one of these: (C, U, l)."

def test_username_mask_with_only_separators():
    person = Person()
    try:
        person.username(mask=".-_")
        assert False
    except ValueError as e:
        assert str(e) == "Username mask must contain at least one of these: (C, U, l)."

def test_username_complex_mask():
    person = Person()
    result = person.username(mask="C_l-d.U")
    assert isinstance(result, str)
    assert len(result) > 0
    assert result.count("_") == 1
    assert result.count("-") == 1
    assert result.count(".") == 1
    parts = result.split(".")
    assert len(parts) == 2
    first_part = parts[0]
    second_part = parts[1]
    subparts = first_part.split("_")
    assert len(subparts) == 2
    assert subparts[0][0].isupper()
    subsubparts = subparts[1].split("-")
    assert len(subsubparts) == 2
    assert subsubparts[0].islower()
    assert subsubparts[1].isdigit()
    assert second_part.isupper()

def test_username_seeded_reproducibility():
    person1 = Person(seed=42)
    result1 = person1.username(mask="C_d")
    person2 = Person(seed=42)
    result2 = person2.username(mask="C_d")
    assert result1 == result2


# LLM-generated content at query #2
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

def test_patronymic_returns_string_when_available():
    person = Person(locale="ru")
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str)

def test_patronymic_returns_none_when_not_available():
    person = Person(locale="en")
    result = person.patronymic(Gender.FEMALE)
    assert result is None


# LLM-generated content at query #3
#--------------------------

def test_patronymic_returns_none_when_patronymics_list_is_empty():
    person = Person()
    person._extract = lambda keys, default: []
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #4
#--------------------------

def test_username_default_mask():
    person = Person()
    result = person.username()
    assert isinstance(result, str)
    assert len(result) > 0

def test_username_custom_mask_capitalized():
    person = Person()
    result = person.username(mask='C')
    assert result[0].isupper()
    assert result[1:].islower()

def test_username_custom_mask_uppercase():
    person = Person()
    result = person.username(mask='U')
    assert result.isupper()

def test_username_custom_mask_lowercase():
    person = Person()
    result = person.username(mask='l')
    assert result.islower()

def test_username_custom_mask_with_digits():
    person = Person()
    result = person.username(mask='d', drange=(2000, 2000))
    assert result == '2000'

def test_username_custom_mask_with_separators():
    person = Person()
    result = person.username(mask='C_C_d', drange=(1999, 1999))
    parts = result.split('_')
    assert len(parts) == 3
    assert parts[0][0].isupper()
    assert parts[0][1:].islower()
    assert parts[1][0].isupper()
    assert parts[1][1:].islower()
    assert parts[2] == '1999'

def test_username_custom_drange():
    person = Person()
    result = person.username(mask='d', drange=(100, 200))
    num = int(result)
    assert 100 <= num <= 200

def test_username_invalid_drange_length():
    person = Person()
    try:
        person.username(drange=(1, 2, 3))
        assert False
    except ValueError:
        assert True

def test_username_mask_without_required_tags():
    person = Person()
    try:
        person.username(mask='d.d')
        assert False
    except ValueError:
        assert True

def test_username_empty_mask():
    person = Person()
    result = person.username(mask='')
    assert isinstance(result, str)

def test_username_mask_with_multiple_separators():
    person = Person()
    result = person.username(mask='C-U.l_d', drange=(2020, 2020))
    assert '2020' in result
    assert '-' in result or '.' in result or '_' in result


# LLM-generated content at query #5
#--------------------------

def test_username_with_uppercase_tag():
    person = Person()
    result = person.username(mask="U")
    assert result.isupper()


# LLM-generated content at query #6
#--------------------------

def test_username_default_mask():
    person = Person()
    result = person.username()
    assert isinstance(result, str)
    assert len(result) > 0

def test_username_custom_mask_capitalized():
    person = Person()
    result = person.username(mask='C')
    assert result[0].isupper()
    assert result[1:].islower()

def test_username_custom_mask_uppercase():
    person = Person()
    result = person.username(mask='U')
    assert result.isupper()

def test_username_custom_mask_lowercase():
    person = Person()
    result = person.username(mask='l')
    assert result.islower()

def test_username_custom_mask_with_digits():
    person = Person()
    result = person.username(mask='d', drange=(2000, 2000))
    assert result == '2000'

def test_username_custom_mask_combined():
    person = Person()
    result = person.username(mask='C_d')
    parts = result.split('_')
    assert len(parts) == 2
    assert parts[0][0].isupper() and parts[0][1:].islower()
    assert parts[1].isdigit()

def test_username_custom_mask_with_separators():
    person = Person()
    result = person.username(mask='C-U.l')
    assert '-' in result
    assert '.' in result

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

def test_username_custom_drange():
    person = Person()
    result = person.username(mask='d', drange=(1995, 1995))
    assert result == '1995'

def test_username_complex_mask():
    person = Person()
    result = person.username(mask='C_C_d')
    parts = result.split('_')
    assert len(parts) == 3
    assert parts[0][0].isupper() and parts[0][1:].islower()
    assert parts[1][0].isupper() and parts[1][1:].islower()
    assert parts[2].isdigit()


# LLM-generated content at query #7
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
    result1 = person.nationality()
    result2 = person.nationality()
    assert result1 != result2

def test_nationality_with_seed_returns_same_value():
    person = Person(seed=42)
    result1 = person.nationality()
    person2 = Person(seed=42)
    result2 = person2.nationality()
    assert result1 == result2


# LLM-generated content at query #8
#--------------------------

def test_nationality_with_gender_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["American"]}
    person.validate_enum = lambda gender, enum_class: "male"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality(gender=Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #9
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


# LLM-generated content at query #10
#--------------------------

def test_patronymic_returns_none_when_no_patronymics():
    person = Person(locale="en")
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #11
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    mock_surnames = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: mock_surnames
    result = person.surname(gender=Gender.MALE)
    assert result in mock_surnames["male"]
    result = person.surname(gender=Gender.FEMALE)
    assert result in mock_surnames["female"]


# LLM-generated content at query #12
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person.validate_enum = lambda gender, enum_class: "male" if gender == Gender.MALE else "female"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.surname(gender=Gender.MALE)
    assert result == "Smith"


# LLM-generated content at query #13
#--------------------------

def test_nationality_with_gender_separated_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["French", "Italian"]}
    person.validate_enum = lambda gender, enum_class: "male" if gender == Gender.MALE else "female"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality(Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #14
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

def test_patronymic_returns_valid_patronymic_for_locale():
    person = Person(locale=Locale.RU)
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str)
    assert len(result) > 0

def test_patronymic_returns_none_for_non_supported_locale():
    person = Person(locale=Locale.EN)
    result = person.patronymic()
    assert result is None

def test_patronymic_uses_random_choice():
    person = Person(locale=Locale.RU)
    results = set()
    for _ in range(10):
        results.add(person.patronymic(Gender.FEMALE))
    assert len(results) > 1 or (len(results) == 1 and next(iter(results)) is None)

def test_patronymic_with_invalid_gender_raises_error():
    person = Person()
    try:
        person.patronymic("invalid")
        assert False
    except NonEnumerableError:
        assert True

def test_patronymic_returns_different_for_male_and_female():
    person = Person(locale=Locale.RU)
    male_result = person.patronymic(Gender.MALE)
    female_result = person.patronymic(Gender.FEMALE)
    assert male_result != female_result or (male_result is None and female_result is None)

def test_patronymic_consistent_with_seed():
    seed = 42
    person1 = Person(seed=seed)
    person2 = Person(seed=seed)
    result1 = person1.patronymic(Gender.MALE)
    result2 = person2.patronymic(Gender.MALE)
    assert result1 == result2

def test_patronymic_extracts_correct_data_key():
    person = Person(locale=Locale.UK)
    result = person.patronymic(Gender.FEMALE)
    assert result is None or isinstance(result, str)


# LLM-generated content at query #15
#--------------------------

def test_nationality_with_dict_nationalities_and_gender():
    mock_extract = lambda keys: {"male": ["Russian", "American"], "female": ["French", "Italian"]}
    mock_validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    mock_random_choice = lambda items: items[0]
    person = Person()
    person._extract = mock_extract
    person.validate_enum = mock_validate_enum
    person.random.choice = mock_random_choice
    result = person.nationality(Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #16
#--------------------------

def test_surname_without_gender():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)
    assert len(result) > 0

def test_surname_with_male_gender():
    person = Person()
    result = person.surname(gender=Gender.MALE)
    assert isinstance(result, str)
    assert len(result) > 0

def test_surname_with_female_gender():
    person = Person()
    result = person.surname(gender=Gender.FEMALE)
    assert isinstance(result, str)
    assert len(result) > 0

def test_surname_with_non_binary_gender():
    person = Person()
    result = person.surname(gender=Gender.NON_BINARY)
    assert isinstance(result, str)
    assert len(result) > 0

def test_surname_with_other_gender():
    person = Person()
    result = person.surname(gender=Gender.OTHER)
    assert isinstance(result, str)
    assert len(result) > 0

def test_surname_returns_different_values():
    person = Person()
    results = set()
    for _ in range(100):
        results.add(person.surname())
    assert len(results) > 1

def test_surname_with_same_seed_returns_same_value():
    person1 = Person(seed=42)
    person2 = Person(seed=42)
    result1 = person1.surname()
    result2 = person2.surname()
    assert result1 == result2

def test_surname_with_different_seed_returns_different_value():
    person1 = Person(seed=42)
    person2 = Person(seed=43)
    result1 = person1.surname()
    result2 = person2.surname()
    assert result1 != result2

def test_surname_with_gender_and_seed():
    person1 = Person(seed=42)
    person2 = Person(seed=42)
    result1 = person1.surname(gender=Gender.MALE)
    result2 = person2.surname(gender=Gender.MALE)
    assert result1 == result2

def test_surname_with_none_gender():
    person = Person()
    result = person.surname(gender=None)
    assert isinstance(result, str)
    assert len(result) > 0


# LLM-generated content at query #17
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person.validate_enum = lambda gender, enum_class: "male" if gender == Gender.MALE else "female"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.surname(gender=Gender.MALE)
    assert result == "Smith"


# LLM-generated content at query #18
#--------------------------

def test_patronymic_with_male_gender():
    person = Person()
    result = person.patronymic(gender=Gender.MALE)
    assert result is None or isinstance(result, str)

def test_patronymic_with_female_gender():
    person = Person()
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None or isinstance(result, str)

def test_patronymic_without_gender():
    person = Person()
    result = person.patronymic()
    assert result is None or isinstance(result, str)

def test_patronymic_with_none_gender():
    person = Person()
    result = person.patronymic(gender=None)
    assert result is None or isinstance(result, str)

def test_patronymic_returns_string_or_none():
    person = Person()
    result = person.patronymic()
    assert result is None or isinstance(result, str)

def test_patronymic_with_invalid_gender():
    person = Person()
    try:
        person.patronymic(gender="invalid")
        assert False
    except Exception:
        assert True

def test_patronymic_uses_random_choice():
    person = Person()
    person.random = MockRandom()
    result = person.patronymic()
    assert result is None or result in person._extract(["patronymic", "male"]) or result in person._extract(["patronymic", "female"])

def test_patronymic_for_locale_without_patronymics():
    person = Person()
    person._extract = lambda keys, default: default
    result = person.patronymic()
    assert result is None

def test_patronymic_for_locale_with_patronymics():
    person = Person()
    person._extract = lambda keys, default: ["Ivanovich", "Petrovich"]
    result = person.patronymic()
    assert result in ["Ivanovich", "Petrovich"]


# LLM-generated content at query #19
#--------------------------

def test_nationality_with_gender_separated_dict():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda value, enum_class: "male" if value == Gender.MALE else "female"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality(gender=Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #20
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

def test_patronymic_locale_specific():
    person = Person(locale=Locale.RU)
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str)

def test_patronymic_locale_specific_female():
    person = Person(locale=Locale.RU)
    result = person.patronymic(Gender.FEMALE)
    assert isinstance(result, str)

def test_patronymic_non_supported_locale():
    person = Person(locale=Locale.EN)
    result = person.patronymic(Gender.MALE)
    assert result is None

def test_patronymic_returns_random_values():
    person = Person(locale=Locale.RU)
    results = {person.patronymic(Gender.MALE) for _ in range(10)}
    assert len(results) > 1

def test_patronymic_with_invalid_gender():
    person = Person()
    try:
        person.patronymic("invalid")
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #21
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    mock_surnames = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: mock_surnames
    result = person.surname(gender=Gender.MALE)
    assert result in mock_surnames["male"]


# LLM-generated content at query #22
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

def test_nationality_with_invalid_gender_raises_error():
    person = Person()
    try:
        person.nationality(gender="INVALID")
        assert False
    except Exception:
        assert True

def test_nationality_different_calls_return_varied():
    person = Person()
    results = [person.nationality() for _ in range(10)]
    assert len(set(results)) > 1


# LLM-generated content at query #23
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

def test_patronymic_returns_string_when_available():
    person = Person(locale="ru")
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str)

def test_patronymic_returns_none_when_not_available():
    person = Person(locale="en")
    result = person.patronymic(Gender.FEMALE)
    assert result is None


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

def test_nationality_with_dict_and_gender():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda gender, enum_class: "male"
    person.random = type('Random', (), {'choice': lambda x: x[0]})()
    result = person.nationality(gender=Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #26
#--------------------------

def test_nationality_with_gender_separated_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["American"]}
    person.validate_enum = lambda gender, enum_class: "male" if gender == Gender.MALE else "female"
    person.random = type('Random', (), {'choice': lambda lst: lst[0]})()
    result = person.nationality(gender=Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #27
#--------------------------

def test_surname_with_gender_separated_dict():
    person = Person()
    mock_surnames = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: mock_surnames
    result = person.surname(gender=Gender.MALE)
    assert result in mock_surnames["male"]
    result = person.surname(gender=Gender.FEMALE)
    assert result in mock_surnames["female"]


# LLM-generated content at query #28
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


