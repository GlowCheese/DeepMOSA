####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
#--------------------------

```python
def test_surname_with_gender_dict():
    person = Person()
    person._extract = lambda keys: {"male": ["Smith"], "female": ["Johnson"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert person.surname(Gender.MALE) == "Smith"


# LLM-generated content at query #3
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


# LLM-generated content at query #4
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
    assert email.endswith("@example.com") or email.endswith("@test.org")

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


# LLM-generated content at query #5
#--------------------------

```python
def test_patronymic_with_gender():
    person = Person()
    gender = Gender.MALE
    result = person.patronymic(gender)
    assert isinstance(result, str) or result is None

def test_patronymic_without_gender():
    person = Person()
    result = person.patronymic()
    assert isinstance(result, str) or result is None

def test_patronymic_with_unsupported_locale():
    person = Person(locale="en_US")
    result = person.patronymic()
    assert result is None


# LLM-generated content at query #6
#--------------------------

```python
def test_person_constructor_initialization():
    person = Person(locale="en_US", seed=42)
    assert person.locale == "en_US"
    assert person.seed == 42


# LLM-generated content at query #7
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics_available():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #8
#--------------------------

```python
def test_email_raises_value_error_when_unique_and_seeded():
    person = Person(seed=42)
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


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
    with pytest.raises(NonEnumerableError):
        person.nationality(gender="invalid_gender")


# LLM-generated content at query #10
#--------------------------

```python
def test_nationality_with_gender_separated_data():
    person = Person()
    person._extract = lambda keys: {"male": ["Russian", "American"], "female": ["Russian", "American"]}
    person.validate_enum = lambda gender, enum: "male" if gender == Gender.MALE else "female"
    person.random.choice = lambda x: x[0]
    assert isinstance(person._extract(["nationality"]), dict)


# LLM-generated content at query #11
#--------------------------

```python
def test_patronymic_returns_none_when_no_patronymics():
    person = Person()
    person._extract = lambda keys, default: default
    assert person.patronymic() is None


# LLM-generated content at query #12
#--------------------------

```python
def test_email_raises_valueerror_when_unique_and_seeded():
    person = Person(seed=42)
    with pytest.raises(ValueError, match="You cannot use «unique» parameter with the seeded provider"):
        person.email(unique=True)


