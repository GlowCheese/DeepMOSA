####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_username_default_mask():
    person = Person()
    username = person.username()
    assert isinstance(username, str)
    assert len(username.split('_')) == 2
    assert username.split('_')[0].islower()
    assert username.split('_')[1].isdigit()

def test_username_custom_mask():
    person = Person()
    username = person.username(mask='C_C_d')
    assert isinstance(username, str)
    assert len(username.split('_')) == 3
    assert username.split('_')[0][0].isupper()
    assert username.split('_')[1][0].isupper()
    assert username.split('_')[2].isdigit()

def test_username_with_drange():
    person = Person()
    username = person.username(mask='l_l_d', drange=(1900, 2021))
    assert isinstance(username, str)
    assert len(username.split('_')) == 3
    assert username.split('_')[0].islower()
    assert username.split('_')[1].islower()
    assert 1900 <= int(username.split('_')[2]) <= 2021

def test_username_invalid_mask():
    person = Person()
    try:
        person.username(mask='123')
        assert False, "Expected ValueError"
    except ValueError:
        assert True

def test_username_invalid_drange():
    person = Person()
    try:
        person.username(drange=(1900, 2021, 2022))
        assert False, "Expected ValueError"
    except ValueError:
        assert True


# LLM-generated content at query #2
#--------------------------

```python
def test_patronymic_returns_none_for_unsupported_locale():
    person = Person(locale="en_US")
    assert person.patronymic() is None

def test_patronymic_returns_valid_patronymic_for_male():
    person = Person(locale="ru_RU")
    patronymic = person.patronymic(gender=Gender.MALE)
    assert isinstance(patronymic, str)
    assert patronymic.endswith(("ович", "евич"))

def test_patronymic_returns_valid_patronymic_for_female():
    person = Person(locale="ru_RU")
    patronymic = person.patronymic(gender=Gender.FEMALE)
    assert isinstance(patronymic, str)
    assert patronymic.endswith(("овна", "евна"))


# LLM-generated content at query #3
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
    with pytest.raises(NonEnumerableError):
        person.nationality(gender="invalid_gender")


# LLM-generated content at query #4
#--------------------------

```python
def test_isinstance_nationalities_dict():
    person = Person()
    nationalities = person._extract(["nationality"])
    assert isinstance(nationalities, dict)


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
    assert any(email.endswith(f"@{domain}") for domain in domains)

def test_email_unique():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2
    assert isinstance(email1, str)
    assert isinstance(email2, str)

def test_email_unique_with_seed_raises():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"


# LLM-generated content at query #6
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #7
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


# LLM-generated content at query #8
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #9
#--------------------------

```python
def test_patronymic_returns_none_when_patronymics_list_is_empty():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #10
#--------------------------

```python
def test_username_default_mask():
    person = Person()
    username = person.username()
    assert isinstance(username, str)
    assert len(username.split('_')) == 2
    assert username.split('_')[0].islower()
    assert username.split('_')[1].isdigit()

def test_username_custom_mask():
    person = Person()
    username = person.username(mask='C_C_d')
    assert isinstance(username, str)
    parts = username.split('_')
    assert len(parts) == 3
    assert parts[0][0].isupper()
    assert parts[1][0].isupper()
    assert parts[2].isdigit()

def test_username_with_digits_range():
    person = Person()
    username = person.username(mask='l_l_d', drange=(1900, 2021))
    assert isinstance(username, str)
    parts = username.split('_')
    assert len(parts) == 3
    assert parts[0].islower()
    assert parts[1].islower()
    assert 1900 <= int(parts[2]) <= 2021

def test_username_invalid_mask():
    person = Person()
    try:
        person.username(mask='#####')
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_username_invalid_drange():
    person = Person()
    try:
        person.username(drange=(1900, 2021, 2022))
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #11
#--------------------------

```python
def test_username_with_capitalized_tag():
    person = Person()
    username = person.username(mask="C")
    assert username == username.capitalize()


# LLM-generated content at query #12
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert person.nationality(gender=Gender.MALE) == "Russian"


# LLM-generated content at query #13
#--------------------------

```python
def test_username_with_uppercase_tag():
    person = Person()
    result = person.username(mask="U")
    assert result.isupper()


# LLM-generated content at query #14
#--------------------------

```python
def test_surname_default():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)
    assert len(result) > 0

def test_surname_with_gender():
    person = Person()
    result = person.surname(Gender.MALE)
    assert isinstance(result, str)
    assert len(result) > 0

def test_surname_with_invalid_gender():
    person = Person()
    try:
        person.surname("invalid_gender")
        assert False, "Expected ValueError for invalid gender"
    except ValueError:
        pass


# LLM-generated content at query #15
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]

    assert person.nationality(gender=Gender.MALE) == "Russian"
    assert person.nationality(gender=Gender.FEMALE) == "French"


# LLM-generated content at query #16
#--------------------------

```python
def test_email_raises_valueerror_when_unique_and_seeded():
    person = Person()
    person._seed = 42  # Simulate a seeded provider
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


# LLM-generated content at query #17
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


# LLM-generated content at query #18
#--------------------------

```python
def test_surname_with_gender_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert person.surname(Gender.MALE) == "Smith"


# LLM-generated content at query #19
#--------------------------

```python
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum: "male"
    person.random.choice = lambda surnames: surnames[0]
    assert person.surname(gender="male") == "Smith"


# LLM-generated content at query #20
#--------------------------

```python
def test_nationality_with_gender_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert person.nationality(gender=Gender.MALE) == "Russian"


# LLM-generated content at query #21
#--------------------------

```python
def test_patronymic_with_valid_gender():
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str) or result is None

def test_patronymic_with_invalid_gender():
    person = Person()
    with pytest.raises(NonEnumerableError):
        person.patronymic("invalid_gender")

def test_patronymic_without_gender():
    person = Person()
    result = person.patronymic()
    assert isinstance(result, str) or result is None

def test_patronymic_with_unsupported_locale():
    person = Person(locale="en_US")
    result = person.patronymic(Gender.MALE)
    assert result is None


# LLM-generated content at query #22
#--------------------------

```python
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert person.surname(Gender.MALE) == "Smith"
    assert person.surname(Gender.FEMALE) == "Johnson"


# LLM-generated content at query #23
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert isinstance(person._extract(["nationality"]), dict)


# LLM-generated content at query #24
#--------------------------

```python
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert person.surname(Gender.MALE) == "Smith"


# LLM-generated content at query #25
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

def test_patronymic_with_none_gender():
    person = Person()
    result = person.patronymic(None)
    assert isinstance(result, str) or result is None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_patronymic_with_gender():
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert isinstance(result, str) or result is None
    assert result in person._extract(["patronymic", "male"], default=[]) or result is None

def test_patronymic_without_gender():
    person = Person()
    result = person.patronymic()
    assert isinstance(result, str) or result is None
    assert result in person._extract(["patronymic", "male"], default=[]) or result is None

def test_patronymic_with_unsupported_locale():
    person = Person(locale="en_US")
    result = person.patronymic(Gender.MALE)
    assert result is None


# LLM-generated content at query #2
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
    assert email.split("@")[1] in domains

def test_email_unique():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed():
    person = Person(seed=42)
    with pytest.raises(ValueError):
        person.email(unique=True)


# LLM-generated content at query #3
#--------------------------

```python
def test_surname_without_gender():
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname.isalpha()

def test_surname_with_gender():
    person = Person()
    surname = person.surname(gender=Gender.MALE)
    assert isinstance(surname, str)
    assert surname.isalpha()

def test_surname_with_invalid_gender():
    person = Person()
    try:
        person.surname(gender="invalid_gender")
    except NonEnumerableError:
        assert True
    else:
        assert False

def test_surname_with_none_gender():
    person = Person()
    surname = person.surname(gender=None)
    assert isinstance(surname, str)
    assert surname.isalpha()


# LLM-generated content at query #4
#--------------------------

```python
def test_email_with_unique_and_seed_raises_value_error():
    person = Person(seed=42)
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


# LLM-generated content at query #5
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #6
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #7
#--------------------------

```python
def test_nationality_without_gender():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)
    assert result

def test_nationality_with_gender():
    person = Person()
    result = person.nationality(gender=Gender.MALE)
    assert isinstance(result, str)
    assert result

def test_nationality_with_invalid_gender():
    person = Person()
    with pytest.raises(NonEnumerableError):
        person.nationality(gender="invalid")


# LLM-generated content at query #8
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #9
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

def test_nationality_with_none_gender():
    person = Person()
    nationality = person.nationality(gender=None)
    assert isinstance(nationality, str)
    assert nationality


# LLM-generated content at query #10
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]

    result = person.nationality(Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #11
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert person.nationality(Gender.MALE) == "Russian"


# LLM-generated content at query #12
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


# LLM-generated content at query #13
#--------------------------

```python
def test_email_with_unique_and_seed_raises_valueerror():
    person = Person(seed=42)
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


# LLM-generated content at query #14
#--------------------------

```python
def test_surname_with_gender_specific_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum_type: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]

    assert person.surname(Gender.MALE) == "Smith"
    assert person.surname(Gender.FEMALE) == "Johnson"


# LLM-generated content at query #15
#--------------------------

```python
def test_email_raises_value_error_when_unique_and_seeded():
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

def test_surname_with_gender():
    person = Person()
    surname = person.surname(Gender.MALE)
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
    person = Person()
    male_surname = person.surname(Gender.MALE)
    female_surname = person.surname(Gender.FEMALE)
    assert isinstance(male_surname, str)
    assert isinstance(female_surname, str)


# LLM-generated content at query #17
#--------------------------

```python
def test_email_with_unique_and_seed_raises_valueerror():
    person = Person(seed=42)
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


# LLM-generated content at query #18
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #19
#--------------------------

```python
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum_type: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda surnames: surnames[0]
    result = person.surname(Gender.MALE)
    assert result == "Smith"


# LLM-generated content at query #20
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys, default=None: {"male": ["Russian"], "female": ["Russian"]}
    person.validate_enum = lambda gender, enum: "male"
    person.random.choice = lambda x: x[0]
    result = person.nationality(gender=Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #21
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


# LLM-generated content at query #22
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert isinstance(person._extract(["nationality"]), dict)


# LLM-generated content at query #23
#--------------------------

```python
def test_email_unique_with_seed_raises_error():
    person = Person(seed=42)
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


# LLM-generated content at query #24
#--------------------------

```python
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert person.surname(Gender.MALE) == "Smith"


# LLM-generated content at query #25
#--------------------------

```python
def test_surname_returns_correct_type_when_surnames_is_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    result = person.surname(Gender.MALE)
    assert isinstance(result, str)


# LLM-generated content at query #26
#--------------------------

```python
def test_email_with_unique_and_seed_raises_valueerror():
    person = Person(seed=42)
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


# LLM-generated content at query #27
#--------------------------

```python
def test_patronymic_returns_none_for_unsupported_locale():
    person = Person(locale="en_US")
    assert person.patronymic() is None

def test_patronymic_returns_none_for_unsupported_gender():
    person = Person(locale="ru_RU")
    assert person.patronymic(Gender.NON_BINARY) is None

def test_patronymic_returns_male_patronymic():
    person = Person(locale="ru_RU")
    patronymic = person.patronymic(Gender.MALE)
    assert isinstance(patronymic, str)
    assert patronymic.endswith(("ович", "евич"))

def test_patronymic_returns_female_patronymic():
    person = Person(locale="ru_RU")
    patronymic = person.patronymic(Gender.FEMALE)
    assert isinstance(patronymic, str)
    assert patronymic.endswith(("овна", "евна"))


# LLM-generated content at query #28
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["French"]}
    person.validate_enum = lambda gender, enum: "male"
    person.random.choice = lambda x: "Russian"
    result = person.nationality(gender=Gender.MALE)
    assert result == "Russian"


# LLM-generated content at query #29
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert person.nationality(Gender.MALE) == "Russian"


# LLM-generated content at query #30
#--------------------------

```python
def test_patronymic_with_valid_gender():
    person = Person()
    result = person.patronymic(gender=Gender.MALE)
    assert isinstance(result, str) or result is None
    if result is not None:
        assert len(result) > 0

def test_patronymic_with_invalid_gender():
    person = Person()
    result = person.patronymic(gender="invalid_gender")
    assert result is None

def test_patronymic_without_gender():
    person = Person()
    result = person.patronymic()
    assert isinstance(result, str) or result is None
    if result is not None:
        assert len(result) > 0


# LLM-generated content at query #31
#--------------------------

```python
def test_surname_with_gender_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person.validate_enum = lambda gender, enum: "male"
    person.random.choice = lambda x: "Smith"
    assert person.surname(gender="male") == "Smith"


# LLM-generated content at query #32
#--------------------------

```python
def test_email_raises_valueerror_when_unique_and_seeded():
    person = Person(locale="en", seed=42)
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


