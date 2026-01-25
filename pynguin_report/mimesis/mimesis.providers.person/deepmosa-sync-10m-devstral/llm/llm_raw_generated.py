####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_nationality_without_gender():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)
    assert len(result) > 0

def test_nationality_with_gender():
    person = Person()
    result = person.nationality(gender=Gender.MALE)
    assert isinstance(result, str)
    assert len(result) > 0

def test_nationality_with_none_gender():
    person = Person()
    result = person.nationality(gender=None)
    assert isinstance(result, str)
    assert len(result) > 0


# LLM-generated content at query #2
#--------------------------

```python
def test_email_with_default_domains():
    person = Person()
    email = person.email()
    assert isinstance(email, str)
    assert "@" in email
    assert "." in email.split("@")[1]

def test_email_with_custom_domains():
    person = Person()
    custom_domains = ["example.com", "test.org"]
    email = person.email(domains=custom_domains)
    assert isinstance(email, str)
    assert any(domain in email for domain in custom_domains)

def test_email_with_unique_and_no_seed():
    person = Person()
    email = person.email(unique=True)
    assert isinstance(email, str)
    assert "@" in email

def test_email_with_unique_and_seed():
    person = Person(seed=42)
    with pytest.raises(ValueError):
        person.email(unique=True)


# LLM-generated content at query #3
#--------------------------

```python
def test_email_default():
    person = Person()
    email = person.email()
    assert isinstance(email, str)
    assert "@" in email
    assert "." in email.split("@")[1]

def test_email_custom_domains():
    person = Person()
    domains = ["example.com", "test.org"]
    email = person.email(domains=domains)
    assert isinstance(email, str)
    assert any(domain in email for domain in domains)

def test_email_unique():
    person = Person()
    email = person.email(unique=True)
    assert isinstance(email, str)
    assert "@" in email
    assert len(email.split("@")[0]) == 32  # UUID hex length

def test_email_unique_with_seed():
    person = Person(seed=42)
    with pytest.raises(ValueError):
        person.email(unique=True)


# LLM-generated content at query #4
#--------------------------

```python
def test_patronymic_returns_none_for_unsupported_locale():
    person = Person(locale="en_US")
    assert person.patronymic() is None

def test_patronymic_returns_valid_name_for_supported_locale():
    person = Person(locale="ru_RU")
    result = person.patronymic()
    assert isinstance(result, str)
    assert len(result) > 0


# LLM-generated content at query #5
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #6
#--------------------------

```python
def test_patronymic_returns_none_when_empty():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #7
#--------------------------

```python
def test_surname_with_gender():
    person = Person()
    result = person.surname(Gender.MALE)
    assert isinstance(result, str)
    assert result.isalpha()

def test_surname_without_gender():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)
    assert result.isalpha()

def test_surname_with_invalid_gender():
    person = Person()
    with pytest.raises(NonEnumerableError):
        person.surname("invalid_gender")


# LLM-generated content at query #8
#--------------------------

```python
def test_email_default():
    person = Person()
    email = person.email()
    assert isinstance(email, str)
    assert "@" in email
    assert "." in email.split("@")[1]

def test_email_custom_domains():
    person = Person()
    domains = ["example.com", "test.org"]
    email = person.email(domains=domains)
    assert isinstance(email, str)
    assert email.endswith(tuple(f"@{domain}" for domain in domains))

def test_email_unique():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed():
    person = Person(seed=42)
    with pytest.raises(ValueError):
        person.email(unique=True)


# LLM-generated content at query #9
#--------------------------

```python
def test_username_with_default_mask():
    person = Person()
    result = person.username()
    assert isinstance(result, str)
    assert len(result.split('_')) == 2
    assert result.split('_')[0].islower()
    assert result.split('_')[1].isdigit()

def test_username_with_custom_mask():
    person = Person()
    result = person.username(mask='C_C_d')
    parts = result.split('_')
    assert len(parts) == 3
    assert parts[0][0].isupper() and parts[0][1:].islower()
    assert parts[1][0].isupper() and parts[1][1:].islower()
    assert parts[2].isdigit()

def test_username_with_uppercase_mask():
    person = Person()
    result = person.username(mask='U.l.d')
    parts = result.split('.')
    assert len(parts) == 3
    assert parts[0].isupper()
    assert parts[1].islower()
    assert parts[2].isdigit()

def test_username_with_invalid_mask():
    person = Person()
    try:
        person.username(mask='123')
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "Username mask must contain at least one of these: (C, U, l)."

def test_username_with_invalid_drange():
    person = Person()
    try:
        person.username(drange=(1900, 2021, 2022))
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "The drange parameter must contain only two integers."


# LLM-generated content at query #10
#--------------------------

```python
def test_nationality_separated_by_gender():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    assert isinstance(person.nationality(gender=Gender.MALE), str)


# LLM-generated content at query #11
#--------------------------

```python
def test_username_with_uppercase_tag():
    person = Person()
    username = person.username(mask="U")
    assert username.isupper()


# LLM-generated content at query #12
#--------------------------

```python
def test_username_with_uppercase_tag():
    person = Person()
    username = person.username(mask="U")
    assert username.isupper()


# LLM-generated content at query #13
#--------------------------

```python
def test_username_default_mask():
    person = Person()
    username = person.username()
    assert len(username.split('_')) == 2
    assert username.split('_')[0].islower()
    assert username.split('_')[1].isdigit()

def test_username_custom_mask():
    person = Person()
    username = person.username(mask='U.l.d')
    parts = username.split('.')
    assert len(parts) == 3
    assert parts[0].isupper()
    assert parts[1].islower()
    assert parts[2].isdigit()

def test_username_with_drange():
    person = Person()
    username = person.username(mask='l_l_d', drange=(1900, 2021))
    parts = username.split('_')
    assert len(parts) == 3
    assert parts[0].islower()
    assert parts[1].islower()
    assert 1900 <= int(parts[2]) <= 2021

def test_username_invalid_drange():
    person = Person()
    try:
        person.username(drange=(1900,))
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_username_missing_required_tags():
    person = Person()
    try:
        person.username(mask='...')
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #14
#--------------------------

```python
def test_nationality_with_gender_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]

    assert isinstance(person._extract(["nationality"]), dict)


# LLM-generated content at query #15
#--------------------------

```python
def test_email_raises_valueerror_when_unique_and_seeded():
    person = Person(seed=42)
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


# LLM-generated content at query #16
#--------------------------

```python
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

def test_surname_consistency():
    person = Person(seed=42)
    surname1 = person.surname()
    surname2 = person.surname()
    assert surname1 == surname2

def test_surname_different_genders():
    person = Person(seed=42)
    male_surname = person.surname(Gender.MALE)
    female_surname = person.surname(Gender.FEMALE)
    assert male_surname != female_surname


# LLM-generated content at query #17
#--------------------------

```python
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]

    assert person.surname(Gender.MALE) == "Smith"
    assert person.surname(Gender.FEMALE) == "Johnson"


# LLM-generated content at query #18
#--------------------------

```python
def test_surname_without_gender():
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname

def test_surname_with_gender():
    person = Person()
    surname = person.surname(gender=Gender.MALE)
    assert isinstance(surname, str)
    assert surname

def test_surname_with_invalid_gender():
    person = Person()
    surname = person.surname(gender="invalid")
    assert isinstance(surname, str)
    assert surname


# LLM-generated content at query #19
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #20
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["Russian"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert isinstance(person._extract(["nationality"]), dict)


# LLM-generated content at query #21
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert isinstance(person._extract(["nationality"]), dict)


# LLM-generated content at query #22
#--------------------------

```python
def test_patronymic_with_valid_gender():
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str) or result is None

def test_patronymic_with_invalid_gender():
    person = Person()
    result = person.patronymic("invalid_gender")
    assert result is None

def test_patronymic_with_no_gender():
    person = Person()
    result = person.patronymic()
    assert isinstance(result, str) or result is None


# LLM-generated content at query #23
#--------------------------

```python
def test_surname_with_gender():
    person = Person()
    result = person.surname(Gender.MALE)
    assert isinstance(result, str)
    assert result

def test_surname_without_gender():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)
    assert result

def test_surname_with_invalid_gender():
    person = Person()
    result = person.surname("invalid_gender")
    assert isinstance(result, str)
    assert result


# LLM-generated content at query #24
#--------------------------

```python
def test_email_with_unique_and_seed():
    person = Person(seed=42)
    try:
        person.email(unique=True)
    except ValueError as e:
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"


# LLM-generated content at query #25
#--------------------------

```python
def test_username_tag_U_uppercase():
    person = Person()
    username = person.username(mask="U")
    assert username.isupper()


# LLM-generated content at query #26
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda gender, enum: "male"
    person.random.choice = lambda x: x[0]
    assert person.nationality(gender=Gender.MALE) == "Russian"


# LLM-generated content at query #27
#--------------------------

```python
def test_patronymic_returns_none_for_unsupported_locale():
    person = Person(locale="en_US")
    assert person.patronymic() is None

def test_patronymic_returns_valid_name_for_supported_locale():
    person = Person(locale="ru_RU")
    assert isinstance(person.patronymic(), str)
    assert len(person.patronymic()) > 0

def test_patronymic_returns_valid_name_for_male_gender():
    person = Person(locale="ru_RU")
    assert isinstance(person.patronymic(Gender.MALE), str)
    assert len(person.patronymic(Gender.MALE)) > 0

def test_patronymic_returns_valid_name_for_female_gender():
    person = Person(locale="ru_RU")
    assert isinstance(person.patronymic(Gender.FEMALE), str)
    assert len(person.patronymic(Gender.FEMALE)) > 0


# LLM-generated content at query #28
#--------------------------

```python
def test_surname_default():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)
    assert result != ""

def test_surname_with_gender():
    person = Person()
    result = person.surname(Gender.MALE)
    assert isinstance(result, str)
    assert result != ""

def test_surname_with_none_gender():
    person = Person()
    result = person.surname(None)
    assert isinstance(result, str)
    assert result != ""


# LLM-generated content at query #29
#--------------------------

```python
def test_username_with_uppercase_tag():
    person = Person()
    username = person.username(mask="U")
    assert username.isupper()


# LLM-generated content at query #30
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_email_default():
    person = Person()
    email = person.email()
    assert isinstance(email, str)
    assert "@" in email
    assert "." in email.split("@")[1]

def test_email_custom_domains():
    person = Person()
    domains = ["example.com", "test.org"]
    email = person.email(domains=domains)
    assert isinstance(email, str)
    assert any(domain in email for domain in domains)

def test_email_unique():
    person = Person()
    email = person.email(unique=True)
    assert isinstance(email, str)
    assert "@" in email
    assert len(email.split("@")[0]) == 32  # UUID4 hex length

def test_email_unique_with_seed():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"


# LLM-generated content at query #2
#--------------------------

```python
def test_email_raises_valueerror_with_unique_and_seed():
    person = Person(seed=42)
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


# LLM-generated content at query #3
#--------------------------

```python
def test_patronymic_returns_none_for_unsupported_locale():
    person = Person(locale="en_US")
    assert person.patronymic() is None

def test_patronymic_returns_valid_patronymic_for_supported_locale():
    person = Person(locale="ru_RU")
    patronymic = person.patronymic()
    assert isinstance(patronymic, str)
    assert len(patronymic) > 0

def test_patronymic_returns_valid_patronymic_for_specific_gender():
    person = Person(locale="ru_RU")
    patronymic = person.patronymic(gender=Gender.MALE)
    assert isinstance(patronymic, str)
    assert len(patronymic) > 0

def test_patronymic_returns_none_for_invalid_gender():
    person = Person(locale="ru_RU")
    assert person.patronymic(gender="invalid") is None


# LLM-generated content at query #4
#--------------------------

```python
def test_surname_default():
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname

def test_surname_with_gender():
    person = Person()
    surname = person.surname(gender=Gender.MALE)
    assert isinstance(surname, str)
    assert surname

def test_surname_with_none_gender():
    person = Person()
    surname = person.surname(gender=None)
    assert isinstance(surname, str)
    assert surname


# LLM-generated content at query #5
#--------------------------

```python
def test_email_default():
    person = Person()
    email = person.email()
    assert isinstance(email, str)
    assert "@" in email
    assert "." in email.split("@")[1]

def test_email_custom_domains():
    person = Person()
    domains = ["example.com", "test.org"]
    email = person.email(domains=domains)
    assert isinstance(email, str)
    assert any(domain in email for domain in domains)

def test_email_unique():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed_raises():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        assert True

def test_email_with_at_domain():
    person = Person()
    domains = ["@example.com", "@test.org"]
    email = person.email(domains=domains)
    assert isinstance(email, str)
    assert any(domain in email for domain in domains)


# LLM-generated content at query #6
#--------------------------

```python
def test_email_default():
    person = Person()
    email = person.email()
    assert isinstance(email, str)
    assert "@" in email
    assert "." in email.split("@")[1]

def test_email_custom_domains():
    person = Person()
    domains = ["example.com", "test.org"]
    email = person.email(domains=domains)
    assert isinstance(email, str)
    assert any(domain in email for domain in domains)

def test_email_unique():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"


# LLM-generated content at query #7
#--------------------------

```python
def test_surname_default():
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname

def test_surname_with_gender():
    person = Person()
    surname_male = person.surname(Gender.MALE)
    surname_female = person.surname(Gender.FEMALE)
    assert isinstance(surname_male, str)
    assert isinstance(surname_female, str)
    assert surname_male
    assert surname_female

def test_surname_with_none_gender():
    person = Person()
    surname = person.surname(None)
    assert isinstance(surname, str)
    assert surname


# LLM-generated content at query #8
#--------------------------

```python
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert person.surname(Gender.MALE) == "Smith"


# LLM-generated content at query #9
#--------------------------

```python
def test_nationality_without_gender():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)
    assert len(result) > 0

def test_nationality_with_gender():
    person = Person()
    result = person.nationality(gender=Gender.MALE)
    assert isinstance(result, str)
    assert len(result) > 0

def test_nationality_with_invalid_gender():
    person = Person()
    result = person.nationality(gender="invalid")
    assert isinstance(result, str)
    assert len(result) > 0


# LLM-generated content at query #10
#--------------------------

```python
def test_nationality_without_gender():
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality

def test_nationality_with_gender():
    person = Person()
    nationality = person.nationality(gender=Gender.MALE)
    assert isinstance(nationality, str)
    assert nationality

def test_nationality_with_invalid_gender():
    person = Person()
    with pytest.raises(NonEnumerableError):
        person.nationality(gender="invalid_gender")


# LLM-generated content at query #11
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert isinstance(person.nationality(Gender.MALE), str)


# LLM-generated content at query #12
#--------------------------

```python
def test_surname_without_gender():
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname

def test_surname_with_valid_gender():
    person = Person()
    surname = person.surname(Gender.MALE)
    assert isinstance(surname, str)
    assert surname

def test_surname_with_invalid_gender():
    person = Person()
    with raises(NonEnumerableError):
        person.surname("invalid_gender")

def test_surname_with_gender_none():
    person = Person()
    surname = person.surname(None)
    assert isinstance(surname, str)
    assert surname


# LLM-generated content at query #13
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #14
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert isinstance(person._extract(["nationality"]), dict)


# LLM-generated content at query #15
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #16
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #17
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert person.nationality(gender=Gender.MALE) == "Russian"


# LLM-generated content at query #18
#--------------------------

```python
def test_patronymic_with_valid_gender():
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str) or result is None

def test_patronymic_without_gender():
    person = Person()
    result = person.patronymic()
    assert isinstance(result, str) or result is None

def test_patronymic_with_invalid_gender():
    person = Person()
    result = person.patronymic("invalid_gender")
    assert isinstance(result, str) or result is None


# LLM-generated content at query #19
#--------------------------

```python
def test_surname_without_gender():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)
    assert result != ""

def test_surname_with_gender():
    person = Person()
    result = person.surname(gender=Gender.MALE)
    assert isinstance(result, str)
    assert result != ""

def test_surname_with_invalid_gender():
    person = Person()
    with pytest.raises(NonEnumerableError):
        person.surname(gender="invalid_gender")


# LLM-generated content at query #20
#--------------------------

```python
def test_email_raises_value_error_when_unique_and_seeded():
    person = Person(seed=42)
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


# LLM-generated content at query #21
#--------------------------

```python
def test_surname_returns_surname_from_dict_when_surnames_is_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum: "male"
    person.random.choice = lambda surnames: surnames[0]
    assert person.surname() == "Smith"


# LLM-generated content at query #22
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics():
    person = Person()
    person._extract = lambda keys, default: []
    assert person.patronymic() is None


# LLM-generated content at query #23
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    result = person.nationality(gender=Gender.MALE)
    assert result in ["Russian", "American"]


# LLM-generated content at query #24
#--------------------------

```python
def test_nationality_without_gender():
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality

def test_nationality_with_gender():
    person = Person()
    nationality = person.nationality(gender=Gender.MALE)
    assert isinstance(nationality, str)
    assert nationality

def test_nationality_with_invalid_gender():
    person = Person()
    with pytest.raises(NonEnumerableError):
        person.nationality(gender="invalid_gender")


