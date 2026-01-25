####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_surname_returns_a_string():
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)

def test_surname_with_gender_returns_a_string():
    person = Person()
    surname = person.surname(Gender.MALE)
    assert isinstance(surname, str)

def test_surname_with_invalid_gender_raises_error():
    person = Person()
    try:
        person.surname("invalid_gender")
        assert False
    except ValueError:
        assert True

def test_surname_returns_non_empty_string():
    person = Person()
    surname = person.surname()
    assert surname != ""

def test_surname_with_gender_returns_non_empty_string():
    person = Person()
    surname = person.surname(Gender.FEMALE)
    assert surname != ""


# LLM-generated content at query #2
#--------------------------

```python
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    result = person.surname(gender=Gender.MALE)
    assert result in ["Smith", "Johnson"]


# LLM-generated content at query #3
#--------------------------

```python
def test_email_default_domain():
    person = Person()
    email = person.email()
    assert isinstance(email, str)
    assert "@" in email

def test_email_custom_domains():
    person = Person()
    custom_domains = ["example.com", "test.org"]
    email = person.email(domains=custom_domains)
    assert isinstance(email, str)
    assert any(domain in email for domain in custom_domains)

def test_email_unique():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed():
    person = Person(seed=123)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError when using unique with seeded provider"
    except ValueError:
        pass

def test_email_no_domains():
    person = Person()
    email = person.email(domains=[])
    assert isinstance(email, str)
    assert "@" in email


# LLM-generated content at query #4
#--------------------------

```python
def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"


# LLM-generated content at query #5
#--------------------------

```python
def test_nationality_with_gender_male():
    person = Person()
    nationality = person.nationality(Gender.MALE)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.MALE]

def test_nationality_with_gender_female():
    person = Person()
    nationality = person.nationality(Gender.FEMALE)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.FEMALE]

def test_nationality_without_gender():
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])

def test_nationality_with_invalid_gender():
    person = Person()
    try:
        nationality = person.nationality("invalid_gender")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        assert True


# LLM-generated content at query #6
#--------------------------

```python
def test_patronymic_with_valid_gender():
    person = Person()
    patronymic = person.patronymic(Gender.MALE)
    assert patronymic is not None

def test_patronymic_with_invalid_gender():
    person = Person()
    patronymic = person.patronymic("INVALID_GENDER")
    assert patronymic is None

def test_patronymic_with_none_gender():
    person = Person()
    patronymic = person.patronymic(None)
    assert patronymic is None

def test_patronymic_with_specific_locale():
    person = Person(locale="ru")
    patronymic = person.patronymic(Gender.FEMALE)
    assert patronymic is not None

def test_patronymic_with_non_specific_locale():
    person = Person(locale="en")
    patronymic = person.patronymic(Gender.MALE)
    assert patronymic is None


# LLM-generated content at query #7
#--------------------------

```
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    result = person.surname(gender=Gender.MALE)
    assert result in ["Smith", "Johnson


# LLM-generated content at query #8
#--------------------------

```python
def test_email_raises_value_error_when_unique_and_seeded():
    person = Person(seed=123)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError not raised"
    except ValueError as e:
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"


# LLM-generated content at query #9
#--------------------------

```python
def test_patronymic_with_valid_gender():
    person = Person()
    patronymic = person.patronymic(Gender.MALE)
    assert patronymic is None or isinstance(patronymic, str)

def test_patronymic_with_none_gender():
    person = Person()
    patronymic = person.patronymic(None)
    assert patronymic is None or isinstance(patronymic, str)


# LLM-generated content at query #10
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person()
    person._extract = lambda keys, default: default
    result = person.patronymic()
    assert result is None


# LLM-generated content at query #11
#--------------------------

```python
def test_surname_returns_string():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)

def test_surname_with_gender_returns_string():
    person = Person()
    result = person.surname(gender=Gender.MALE)
    assert isinstance(result, str)

def test_surname_with_invalid_gender_raises_error():
    person = Person()
    try:
        person.surname(gender="invalid")
        assert False
    except NonEnumerableError:
        assert True

def test_surname_returns_different_values():
    person = Person()
    result1 = person.surname()
    result2 = person.surname()
    assert result1 != result2

def test_surname_with_seed_returns_same_value():
    person = Person(seed=42)
    result1 = person.surname()
    person = Person(seed=42)
    result2 = person.surname()
    assert result1 == result2


# LLM-generated content at query #12
#--------------------------

```
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person()
    person._extract = lambda keys, default: []
    result = person.patronymic()
    assert result is None


# LLM-generated content at query #13
#--------------------------

```python
def test_email_generates_random_email():
    person = Person()
    email = person.email()
    assert "@" in email
    assert "." in email.split("@")[1]

def test_email_uses_custom_domains():
    person = Person()
    custom_domains = ["example.com", "test.org"]
    email = person.email(domains=custom_domains)
    assert email.endswith(tuple(custom_domains))

def test_email_generates_unique_emails():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_raises_error_when_unique_and_seeded():
    person = Person(seed=123)
    try:
        person.email(unique=True)
        assert False
    except ValueError:
        assert True


# LLM-generated content at query #14
#--------------------------

```python
def test_nationality_with_gender_separated_dict():
    person = Person()
    nationalities = {"male": ["Russian", "American"], "female": ["French", "German"]}
    person._extract = lambda keys: nationalities
    gender = Gender.MALE
    result = person.nationality(gender)
    assert result in nationalities["male"]


# LLM-generated content at query #15
#--------------------------

```python
def test_nationality_with_gender_separated_dict():
    person = Person()
    nationalities = {"Male": ["Russian", "American"], "Female": ["French", "German"]}
    person._extract = lambda _: nationalities
    result = person.nationality(Gender.Male)
    assert result in ["Russian", "American"]

def test_nationality_without_gender_separated_dict():
    person = Person()
    nationalities = ["Russian", "American", "French", "German"]
    person._extract = lambda _: nationalities
    result = person.nationality()
    assert result in nationalities


# LLM-generated content at query #16
#--------------------------

```
def test_nationality_without_gender():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)
    assert len(result) > 0

def test_nationality_with_male_gender():
    person = Person()
    result = person.nationality(Gender.MALE)
    assert isinstance(result, str)
    assert len(result) > 0

def test_nationality_with_female_gender():
    person = Person()
    result = person.nationality(Gender.FEMALE)
    assert isinstance(result, str)
    assert len(result) > 0

def test_nationality_with_other_gender():
    person = Person()
    result = person.nationality(Gender.OTHER)
    assert isinstance(result, str)
    assert len(result) > 0


# LLM-generated content at query #17
#--------------------------

```python
def test_nationality_with_gender_dict():
    person = Person()
    nationalities = {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person._extract = lambda keys: nationalities
    result = person.nationality(Gender.MALE)
    assert result in nationalities["male"]


# LLM-generated content at query #18
#--------------------------

```python
def test_patronymic_with_gender_male():
    person = Person()
    patronymic = person.patronymic(gender=Gender.MALE)
    assert patronymic is None or isinstance(patronymic, str)

def test_patronymic_with_gender_female():
    person = Person()
    patronymic = person.patronymic(gender=Gender.FEMALE)
    assert patronymic is None or isinstance(patronymic, str)

def test_patronymic_without_gender():
    person = Person()
    patronymic = person.patronymic()
    assert patronymic is None or isinstance(patronymic, str)


# LLM-generated content at query #19
#--------------------------

def test_surname_with_gender_separated_surnames():
    person = Person()
    surnames_dict = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: surnames_dict
    surname = person.surname(gender=Gender.MALE)
    assert surname in surnames_dict["male"]


# LLM-generated content at query #20
#--------------------------

```python
def test_email_with_default_domain():
    person = Person()
    email = person.email()
    assert "@" in email
    assert email.endswith(".com") or email.endswith(".net") or email.endswith(".org")

def test_email_with_custom_domain():
    person = Person()
    custom_domains = ["@example.com", "@test.org"]
    email = person.email(domains=custom_domains)
    assert any(email.endswith(domain) for domain in custom_domains)

def test_email_unique():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed_raises_error():
    person = Person(seed=123)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        assert True

def test_email_with_empty_domains():
    person = Person()
    email = person.email(domains=[])
    assert "@" in email
    assert email.endswith(".com") or email.endswith(".net") or email.endswith(".org")

def test_email_with_single_custom_domain():
    person = Person()
    custom_domain = ["@custom.com"]
    email = person.email(domains=custom_domain)
    assert email.endswith(custom_domain[0])


# LLM-generated content at query #21
#--------------------------

```python
def test_nationality_with_dict_input():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["American"]}
    result = person.nationality(Gender.MALE)
    assert result in ["Russian", "American"]


# LLM-generated content at query #22
#--------------------------

```python
def test_patronymic_with_valid_gender():
    person = Person()
    patronymic_male = person.patronymic(Gender.MALE)
    patronymic_female = person.patronymic(Gender.FEMALE)
    assert patronymic_male is not None or patronymic_female is not None

def test_patronymic_with_invalid_gender():
    person = Person()
    patronymic = person.patronymic()
    assert patronymic is None

def test_patronymic_with_none_gender():
    person = Person()
    patronymic = person.patronymic(None)
    assert patronymic is None


# LLM-generated content at query #23
#--------------------------

```
def test_surname_returns_string():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)

def test_surname_with_gender_returns_string():
    person = Person()
    result = person.surname(Gender.MALE)
    assert isinstance(result, str)

def test_surname_with_invalid_gender_raises_error():
    person = Person()
    try:
        person.surname("invalid_gender")
        assert False
    except NonEnumerableError:
        assert True

def test_surname_returns_different_values():
    person = Person()
    result1 = person.surname()
    result2 = person.surname()
    assert result1 != result2

def test_surname_with_same_seed_returns_same_value():
    person1 = Person(seed=42)
    person2 = Person(seed=42)
    assert person1.surname() == person2.surname


# LLM-generated content at query #24
#--------------------------

```python
def test_email_raises_value_error_when_unique_and_seeded():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError not raised"
    except ValueError as e:
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"


# LLM-generated content at query #25
#--------------------------

```python
def test_email_default_domain():
    person = Person()
    email = person.email()
    assert "@" in email
    assert "." in email
    assert len(email.split("@")[0]) > 0

def test_email_custom_domain():
    person = Person()
    custom_domains = ["example.com", "test.org"]
    email = person.email(domains=custom_domains)
    assert any(domain in email for domain in custom_domains)

def test_email_unique():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed_raises_error():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        assert True

def test_email_default_domain_starts_with_at():
    person = Person()
    email = person.email(domains=["@example.com"])
    assert email.endswith("@example.com")

def test_email_non_default_domain():
    person = Person()
    email = person.email(domains=["test.com"])
    assert email.endswith("@test.com")


# LLM-generated content at query #26
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person(locale="en")
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #27
#--------------------------

```python
def test_surname_with_gender_dict():
    person = Person()
    surnames_dict = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: surnames_dict
    surname_male = person.surname(gender=Gender.MALE)
    surname_female = person.surname(gender=Gender.FEMALE)
    assert surname_male in surnames_dict["male"]
    assert surname_female in surnames_dict["female"]


# LLM-generated content at query #28
#--------------------------

```python
def test_nationality_default_gender():
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)

def test_nationality_male_gender():
    person = Person()
    nationality = person.nationality(Gender.MALE)
    assert isinstance(nationality, str)

def test_nationality_female_gender():
    person = Person()
    nationality = person.nationality(Gender.FEMALE)
    assert isinstance(nationality, str)

def test_nationality_invalid_gender():
    person = Person()
    nationality = person.nationality("invalid_gender")
    assert isinstance(nationality, str)


# LLM-generated content at query #29
#--------------------------

```python
def test_email_with_default_domain():
    person = Person()
    email = person.email()
    assert "@" in email
    assert "." in email

def test_email_with_custom_domain():
    person = Person()
    email = person.email(domains=["example.com"])
    assert email.endswith("@example.com")

def test_email_unique():
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

def test_email_unique_with_seed_raises_value_error():
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_email_with_empty_domains_raises_value_error():
    person = Person()
    try:
        person.email(domains=[])
        assert False, "Expected ValueError"
    except ValueError:
        pass


# LLM-generated content at query #30
#--------------------------

```python
def test_surname_with_gender_separated_surnames():
    person = Person()
    surnames = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: surnames
    surname = person.surname(gender=Gender.MALE)
    assert surname in surnames["male"]


# LLM-generated content at query #31
#--------------------------

```python
def test_nationality_gender_separation():
    person = Person()
    nationalities = {"male": ["Russian", "American"], "female": ["French", "German"]}
    person._extract = lambda keys: nationalities
    result_male = person.nationality(gender=Gender.MALE)
    result_female = person.nationality(gender=Gender.FEMALE)
    assert result_male in nationalities["male"]
    assert result_female in nationalities["female"]


# LLM-generated content at query #32
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person(locale="EN")
    assert person.patronymic() is None


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_nationality_with_gender():
    person = Person()
    nationality_male = person.nationality(Gender.MALE)
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_male, str)
    assert isinstance(nationality_female, str)

def test_nationality_without_gender():
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)

def test_nationality_with_invalid_gender():
    person = Person()
    try:
        person.nationality("INVALID_GENDER")
    except Exception as e:
        assert isinstance(e, ValueError)


# LLM-generated content at query #2
#--------------------------

```python
def test_nationality_with_gender_separated():
    person = Person()
    nationalities = {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person._extract = lambda keys: nationalities
    result = person.nationality(gender=Gender.MALE)
    assert result in nationalities["male"]


# LLM-generated content at query #3
#--------------------------

```
def test_username_default_mask():
    person = Person()
    username = person.username()
    assert isinstance(username, str)
    assert len(username) > 0
    assert any(c.isalpha() for c in username)
    assert any(c.isdigit() for c in username)

def test_username_custom_mask():
    person = Person()
    username = person.username(mask='C_C_d')
    parts = username.split('_')
    assert len(parts) == 3
    assert parts[0][0].isupper()
    assert parts[1][0].isupper()
    assert parts[2].isdigit()

def test_username_drange():
    person = Person()
    username = person.username(drange=(2000, 2020))
    year_part = username.split('_')[-1]
    assert 2000 <= int(year_part) <= 2020

def test_username_invalid_drange():
    person = Person()
    try:
        person.username(drange=(2000,))
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for invalid drange"

def test_username_invalid_mask():
    person = Person()
    try:
        person.username(mask='invalid')
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for invalid mask"

def test_username_no_alpha_tags():
    person = Person()
    try:
        person.username(mask='d.d.d')
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for mask without alpha tags"


# LLM-generated content at query #4
#--------------------------

```python
def test_patronymic_with_valid_gender():
    person = Person()
    patronymic = person.patronymic(gender=Gender.MALE)
    assert patronymic is not None

def test_patronymic_with_invalid_gender():
    person = Person()
    patronymic = person.patronymic(gender="INVALID_GENDER")
    assert patronymic is None

def test_patronymic_without_gender():
    person = Person()
    patronymic = person.patronymic()
    assert patronymic is not None

def test_patronymic_with_non_ru_uk_locale():
    person = Person(locale="en")
    patronymic = person.patronymic(gender=Gender.FEMALE)
    assert patronymic is None


# LLM-generated content at query #5
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person()
    patronymic = person.patronymic()
    assert patronymic is None


# LLM-generated content at query #6
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person(locale="en")
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #7
#--------------------------

```
def test_surname_returns_string():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)

def test_surname_with_gender_returns_string():
    person = Person()
    result = person.surname(gender=Gender.MALE)
    assert isinstance(result, str)

def test_surname_with_invalid_gender_raises_error():
    person = Person()
    try:
        person.surname(gender="invalid")
        assert False
    except NonEnumerableError:
        assert True

def test_surname_returns_different_values():
    person = Person()
    result1 = person.surname()
    result2 = person.surname()
    assert result1 != result2

def test_surname_with_same_seed_returns_same_value():
    person = Person(seed=42)
    result1 = person.surname()
    person = Person(seed=42)
    result2 = person.surname()
    assert result1 == result2


# LLM-generated content at query #8
#--------------------------

```python
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    result = person.surname(gender=Gender.MALE)
    assert result in ["Smith", "Johnson"]


# LLM-generated content at query #9
#--------------------------

def test_nationality_with_gender_dict():
    class MockPerson(Person):
        def _extract(self, keys, default=None):
            return {"male": ["Russian"], "female": ["French"]}

    person = MockPerson()
    result = person.nationality(gender=Gender.MALE)
    assert result in ["Russian"]


# LLM-generated content at query #10
#--------------------------

```
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    result = person.surname(gender=Gender.MALE)
    assert result in ["Smith", "Johnson


# LLM-generated content at query #11
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person(locale="en")
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #12
#--------------------------

```python
def test_surname_with_gender_separated_dict():
    person = Person()
    surnames_dict = {"male": ["Smith", "Johnson"], "female": ["Williams", "Brown"]}
    person._extract = lambda keys: surnames_dict
    surname = person.surname(gender=Gender.MALE)
    assert surname in surnames_dict["male"]


# LLM-generated content at query #13
#--------------------------

```python
def test_nationality_returns_string():
    person = Person()
    result = person.nationality()
    assert isinstance(result, str)

def test_nationality_with_gender_returns_string():
    person = Person()
    result_male = person.nationality(gender=Gender.MALE)
    result_female = person.nationality(gender=Gender.FEMALE)
    assert isinstance(result_male, str)
    assert isinstance(result_female, str)

def test_nationality_with_invalid_gender_raises_error():
    person = Person()
    try:
        person.nationality(gender="invalid")
        assert False
    except NonEnumerableError:
        assert True

def test_nationality_values_are_from_data():
    person = Person()
    nationalities = person._extract(["nationality"])
    if isinstance(nationalities, dict):
        nationalities = list(nationalities.values())
        nationalities = [item for sublist in nationalities for item in sublist]
    result = person.nationality()
    assert result in nationalities


# LLM-generated content at query #14
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person()
    person._extract = lambda keys, default: []
    result = person.patronymic(Gender.MALE)
    assert result is None


# LLM-generated content at query #15
#--------------------------

```python
def test_surname_with_gender_separated_surnames():
    person = Person()
    person._extract = lambda _: {"male": ["Smith"], "female": ["Johnson"]}
    result = person.surname(gender=Gender.MALE)
    assert result in ["Smith", "Johnson"]


# LLM-generated content at query #16
#--------------------------

```python
def test_nationality_returns_string_when_nationalities_is_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["American"]}
    result = person.nationality(gender=Gender.MALE)
    assert isinstance(result, str)


# LLM-generated content at query #17
#--------------------------

```python
def test_surname_returns_string():
    person = Person()
    result = person.surname()
    assert isinstance(result, str)

def test_surname_with_gender_returns_string():
    person = Person()
    result = person.surname(gender=Gender.MALE)
    assert isinstance(result, str)

def test_surname_with_invalid_gender_raises_error():
    person = Person()
    try:
        person.surname(gender="invalid")
        assert False
    except NonEnumerableError:
        assert True

def test_surname_returns_different_values():
    person = Person()
    result1 = person.surname()
    result2 = person.surname()
    assert result1 != result2

def test_surname_with_seed_returns_same_value():
    person = Person(seed=42)
    result1 = person.surname()
    result2 = person.surname()
    assert result1 == result2


# LLM-generated content at query #18
#--------------------------

```python
def test_patronymic_returns_none_for_non_supported_locale():
    person = Person(locale="en")
    result = person.patronymic()
    assert result is None

def test_patronymic_returns_string_for_supported_locale():
    person = Person(locale="ru")
    result = person.patronymic()
    assert isinstance(result, str)

def test_patronymic_returns_gender_specific_result():
    person = Person(locale="ru")
    male_result = person.patronymic(Gender.MALE)
    female_result = person.patronymic(Gender.FEMALE)
    assert male_result != female_result
    assert isinstance(male_result, str)
    assert isinstance(female_result, str)

def test_patronymic_returns_random_results():
    person = Person(locale="ru")
    results = {person.patronymic(Gender.MALE) for _ in range(10)}
    assert len(results) > 1

def test_patronymic_handles_none_gender():
    person = Person(locale="ru")
    result = person.patronymic(None)
    assert isinstance(result, str)


# LLM-generated content at query #19
#--------------------------

```
def test_nationality_with_gender_specific_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["French"]}
    result = person.nationality(gender=Gender.MALE)
    assert result in ["Russian"]


# LLM-generated content at query #20
#--------------------------

```
def test_nationality_with_gender_separated_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["American"]}
    result = person.nationality(gender=Gender.MALE)
    assert result in ["Russian", "American"]


# LLM-generated content at query #21
#--------------------------

```python
def test_surname_method_generates_random_surname():
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert len(surname) > 0

def test_surname_method_generates_surname_for_specified_gender():
    person = Person()
    surname_male = person.surname(gender=Gender.MALE)
    surname_female = person.surname(gender=Gender.FEMALE)
    assert isinstance(surname_male, str)
    assert isinstance(surname_female, str)
    assert len(surname_male) > 0
    assert len(surname_female) > 0

def test_surname_method_generates_surname_for_default_gender():
    person = Person()
    surname_default = person.surname()
    assert isinstance(surname_default, str)
    assert len(surname_default) > 0


# LLM-generated content at query #22
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person(locale="en")
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #23
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person(locale="en")
    result = person.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #24
#--------------------------

```python
def test_surname_returns_string():
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)

def test_surname_with_gender_returns_string():
    person = Person()
    surname = person.surname(gender=Gender.MALE)
    assert isinstance(surname, str)

def test_surname_with_invalid_gender_raises_error():
    person = Person()
    try:
        person.surname(gender="INVALID")
        assert False
    except NonEnumerableError:
        assert True

def test_surname_returns_non_empty_string():
    person = Person()
    surname = person.surname()
    assert surname != ""

def test_surname_with_gender_returns_non_empty_string():
    person = Person()
    surname = person.surname(gender=Gender.FEMALE)
    assert surname != ""


# LLM-generated content at query #25
#--------------------------

```
def test_nationality_with_gender_specific_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian"], "female": ["French"]}
    result = person.nationality(gender=Gender.MALE)
    assert result in ["Russian


