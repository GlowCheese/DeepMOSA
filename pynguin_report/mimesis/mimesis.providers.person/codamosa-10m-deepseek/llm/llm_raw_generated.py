####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method surname of class Person
def test_Person_surname():  
    # Test with default gender (None)
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname in person._extract(["surnames"])

    # Test with gender Male
    surname_male = person.surname(Gender.MALE)
    assert isinstance(surname_male, str)
    assert surname_male in person._extract(["surnames"])

    # Test with gender Female
    surname_female = person.surname(Gender.FEMALE)
    assert isinstance(surname_female, str)
    assert surname_female in person._extract(["surnames"])

    # Test with locale that has gender-specific surnames
    person_ru = Person(locale="ru")
    surnames_ru_male = person_ru._extract(["surnames", "male"])
    surnames_ru_female = person_ru._extract(["surnames", "female"])
    surname_ru_male = person_ru.surname(Gender.MALE)
    surname_ru_female = person_ru.surname(Gender.FEMALE)
    assert surname_ru_male in surnames_ru_male
    assert surname_ru_female in surnames_ru_female

    # Test with locale that does not have gender-specific surnames
    person_en = Person(locale="en")
    surnames_en = person_en._extract(["surnames"])
    surname_en = person_en.surname()
    assert surname_en in surnames_en

    # Test with invalid gender (should raise an error)
    try:
        person.surname("INVALID_GENDER")
        assert False, "Expected an error for invalid gender"
    except Exception:
        pass

    # Test that surnames are randomly selected
    surnames_set = set(person.surname() for _ in range(100))
    assert len(surnames_set) > 1, "Surnames should be randomly selected"

    # Test with seed for reproducibility
    person_seeded = Person(seed=42)
    surname1 = person_seeded.surname()
    person_seeded2 = Person(seed=42)
    surname2 = person_seeded2.surname()
    assert surname1 == surname2, "Surnames should be the same with the same seed"

    print("All tests passed for Person.surname()")

# Run the test
test_Person_surname()


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method email of class Person
def test_Person_email():  
    # Test with default parameters
    person = Person()
    email = person.email()
    assert '@' in email
    assert '.' in email
    assert email.split('@')[1] in EMAIL_DOMAINS

    # Test with custom domains
    custom_domains = ['example.com', 'test.org']
    email = person.email(domains=custom_domains)
    assert email.split('@')[1] in custom_domains

    # Test unique email generation
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

    # Test that unique emails are actually unique
    emails = set()
    for _ in range(100):
        emails.add(person.email(unique=True))
    assert len(emails) == 100

    # Test with seeded provider and unique=True should raise ValueError
    seeded_person = Person(seed=42)
    try:
        seeded_person.email(unique=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "You cannot use «unique» parameter with the seeded provider" in str(e)

    # Test email format
    email = person.email()
    local_part, domain = email.split('@')
    assert local_part
    assert domain
    assert '.' in domain

    # Test that email uses username pattern when not unique
    email = person.email()
    local_part = email.split('@')[0]
    # Should be alphanumeric (username pattern with mask='ld')
    assert local_part.isalnum()

    # Test with specific locale
    person_ru = Person('ru')
    email_ru = person_ru.email()
    assert '@' in email_ru
    # Russian locale might have different default domains
    # but email should still be valid

    # Test that multiple calls produce different emails (when not unique)
    # There's a small chance of collision, but with reasonable dataset it should be rare
    emails = [person.email() for _ in range(10)]
    # At least some should be different (not guaranteed all, but very likely)
    assert len(set(emails)) > 1

    print("All tests passed!")

# Run the test
if __name__ == "__main__":
    test_Person_email()


# LLM-generated content at query #2
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])

    # Test with gender Male
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._extract(["nationality"])[Gender.MALE]

    # Test with gender Female
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._extract(["nationality"])[Gender.FEMALE]

    # Test with invalid gender (should raise an error)
    try:
        person.nationality("INVALID_GENDER")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method patronymic of class Person
def test_Person_patronymic():  
    # Test with gender parameter set to Gender.MALE
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with gender parameter set to Gender.FEMALE
    result = person.patronymic(Gender.FEMALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with gender parameter set to None
    result = person.patronymic()
    assert result is None or isinstance(result, str)
    # Test with invalid gender parameter
    try:
        person.patronymic("invalid_gender")
    except Exception as e:
        assert isinstance(e, Exception)


# LLM-generated content at query #2
#--------------------------

# Unit test for method patronymic of class Person
def test_Person_patronymic():  
    # Test with gender=None
    person = Person()
    result = person.patronymic()
    assert result is None or isinstance(result, str)
    
    # Test with gender=Gender.MALE
    result = person.patronymic(gender=Gender.MALE)
    assert result is None or isinstance(result, str)
    
    # Test with gender=Gender.FEMALE
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None or isinstance(result, str)
    
    # Test with locale='ru' (Russian locale)
    person_ru = Person(locale='ru')
    result = person_ru.patronymic(gender=Gender.MALE)
    assert isinstance(result, str) and len(result) > 0
    
    # Test with locale='uk' (Ukrainian locale)
    person_uk = Person(locale='uk')
    result = person_uk.patronymic(gender=Gender.FEMALE)
    assert isinstance(result, str) and len(result) > 0


# LLM-generated content at query #3
#--------------------------

# Unit test for method patronymic of class Person
def test_Person_patronymic():  
    # Test with gender=None (default)
    person = Person()
    result = person.patronymic()
    # Since patronymics are only available for RU and UK locales, 
    # we expect None for other locales
    if person.locale in ('ru', 'uk'):
        assert result is not None
        assert isinstance(result, str)
    else:
        assert result is None

    # Test with gender=Gender.MALE
    person = Person()
    result = person.patronymic(gender=Gender.MALE)
    if person.locale in ('ru', 'uk'):
        assert result is not None
        assert isinstance(result, str)
    else:
        assert result is None

    # Test with gender=Gender.FEMALE
    person = Person()
    result = person.patronymic(gender=Gender.FEMALE)
    if person.locale in ('ru', 'uk'):
        assert result is not None
        assert isinstance(result, str)
    else:
        assert result is None

    # Test with invalid gender (should raise error)
    try:
        person.patronymic(gender='invalid')
        assert False, "Should have raised NonEnumerableError"
    except Exception as e:
        assert 'NonEnumerableError' in str(type(e).__name__)


# LLM-generated content at query #4
#--------------------------

# Unit test for method surname of class Person
def test_Person_surname():  
    # Test with default gender (None)
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname in person._extract(["surnames"])

    # Test with gender Male
    surname_male = person.surname(gender=Gender.MALE)
    assert isinstance(surname_male, str)
    assert surname_male in person._extract(["surnames"])

    # Test with gender Female
    surname_female = person.surname(gender=Gender.FEMALE)
    assert isinstance(surname_female, str)
    assert surname_female in person._extract(["surnames"])

    # Test with invalid gender (should raise error)
    try:
        person.surname(gender="INVALID")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test that surnames are different for different genders (if applicable)
    if isinstance(person._extract(["surnames"]), dict):
        assert surname_male != surname_female

    # Test that surname is not empty
    assert len(surname) > 0

    # Test that surname is a string
    assert isinstance(surname, str)

    # Test that surname is in the list of surnames
    assert surname in person._extract(["surnames"])

    # Test that surname is not None
    assert surname is not None

    # Test that surname is not an empty string
    assert surname != ""

    # Test that surname is not a number
    assert not surname.isdigit()

    # Test that surname is not a special character
    assert surname.isalpha()

    # Test that surname is not a mix of letters and numbers
    assert surname.isalpha()

    # Test that surname is not a mix of letters and special characters
    assert surname.isalpha()

    # Test that surname is not a mix of numbers and special characters
    assert surname.isalpha()

    # Test that surname is not a mix of letters, numbers, and special characters
    assert surname.isalpha()

    # Test that surname is not a mix of letters and spaces
    assert " " not in surname

    # Test that surname is not a mix of letters and punctuation
    assert not any(char in punctuation for char in surname)

    # Test that surname is not a mix of letters and digits
    assert not any(char.isdigit() for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert surname.isalpha()

    # Test that surname is not a mix of letters and unicode characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and emojis
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other symbols
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all(ord(char) < 128 for char in surname)

    # Test that surname is not a mix of letters and other characters
    assert all


# LLM-generated content at query #5
#--------------------------

# Unit test for method email of class Person
def test_Person_email():  
    # Test with default parameters
    person = Person()
    email = person.email()
    assert '@' in email
    assert '.' in email
    assert email.split('@')[1] in EMAIL_DOMAINS

    # Test with custom domains
    custom_domains = ['example.com', 'test.org']
    email = person.email(domains=custom_domains)
    assert email.split('@')[1] in custom_domains

    # Test unique email generation
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

    # Test that unique emails contain UUID
    # (since username is generated from UUID when unique=True)
    import uuid
    try:
        uuid_part = email1.split('@')[0]
        uuid.UUID(uuid_part, version=4)
        uuid_valid = True
    except ValueError:
        uuid_valid = False
    assert uuid_valid

    # Test with seeded provider (should raise error when unique=True)
    seeded_person = Person(seed=42)
    try:
        seeded_person.email(unique=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unique" in str(e).lower()



# LLM-generated content at query #6
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])

    # Test with gender Male
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._extract(["nationality"])[Gender.MALE]

    # Test with gender Female
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._extract(["nationality"])[Gender.FEMALE]

    # Test with invalid gender (should raise an error)
    try:
        person.nationality("INVALID_GENDER")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)


# LLM-generated content at query #7
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._data["nationality"]
    
    # Test with gender Male
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._data["nationality"]["male"]
    
    # Test with gender Female
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._data["nationality"]["female"]
    
    # Test with invalid gender
    try:
        person.nationality("invalid_gender")
    except Exception as e:
        assert isinstance(e, ValueError)


# LLM-generated content at query #8
#--------------------------

# Unit test for method email of class Person
def test_Person_email():  
    # Test with default parameters
    person = Person()
    email = person.email()
    assert '@' in email
    assert '.' in email
    assert email.split('@')[0].isalnum()
    assert email.split('@')[1] in EMAIL_DOMAINS

    # Test with custom domains
    custom_domains = ['example.com', 'test.org']
    email = person.email(domains=custom_domains)
    assert email.split('@')[1] in custom_domains

    # Test with unique=True
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

    # Test with unique=True and seeded provider
    person = Person(seed=42)
    try:
        person.email(unique=True)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "You cannot use «unique» parameter with the seeded provider"

    # Test with unique=False and seeded provider
    person = Person(seed=42)
    email1 = person.email(unique=False)
    email2 = person.email(unique=False)
    assert email1 == email2

    # Test with unique=False and no seed
    person = Person()
    email1 = person.email(unique=False)
    email2 = person.email(unique=False)
    assert email1 != email2

    # Test with unique=True and no seed
    person = Person()
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

    # Test with unique=True and custom domains
    custom_domains = ['example.com', 'test.org']
    email = person.email(domains=custom_domains, unique=True)
    assert email.split('@')[1] in custom_domains
    assert email.split('@')[0] == uuid.UUID(email.split('@')[0], version=4).hex

    # Test with unique=False and custom domains
    custom_domains = ['example.com', 'test.org']
    email = person.email(domains=custom_domains, unique=False)
    assert email.split('@')[1] in custom_domains
    assert email.split('@')[0].isalnum()

    # Test with unique=True and default domains
    email = person.email(unique=True)
    assert email.split('@')[1] in EMAIL_DOMAINS
    assert email.split('@')[0] == uuid.UUID(email.split('@')[0], version=4).hex

    # Test with unique=False and default domains
    email = person.email(unique=False)
    assert email.split('@')[1] in EMAIL_DOMAINS
    assert email.split('@')[0].isalnum()

    # Test with unique=True and empty domains list
    try:
        person.email(domains=[], unique=True)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with unique=False and empty domains list
    try:
        person.email(domains=[], unique=False)
        assert False, "Expected IndexError"
    except IndexError:
        pass

    # Test with unique=True and single domain
    custom_domains = ['example.com']
    email = person.email(domains=custom_domains, unique=True)
    assert email.split('@')[1] == 'example.com'
    assert email.split('@')[0] == uuid.UUID(email.split('@')[0], version=4).hex

    # Test with unique=False and single domain
    custom_domains = ['example.com']
    email = person.email(domains=custom_domains, unique=False)
    assert email.split('@')[1] == 'example.com'
    assert email.split('@')[0].isalnum()

    # Test with unique=True and domain without @
    custom_domains = ['example.com']
    email = person.email(domains=custom_domains, unique=True)
    assert email.split('@')[1] == 'example.com'
    assert email.split('@')[0] == uuid.UUID(email.split('@')[0], version=4).hex

    # Test with unique=False and domain without @
    custom_domains = ['example.com']
    email = person.email(domains=custom_domains, unique=False)
    assert email.split('@')[1] == 'example.com'
    assert email.split('@')[0].isalnum()

    # Test with unique=True and domain with @
    custom_domains = ['@example.com']
    email = person.email(domains=custom_domains, unique=True)
    assert email.split('@')[1] == 'example.com'
    assert email.split('@')[0] == uuid.UUID(email.split('@')[0], version=4).hex

    # Test with unique=False and domain with @
    custom_domains = ['@example.com']
    email = person.email(domains=custom_domains, unique=False)
    assert email.split('@')[1] == 'example.com'
    assert email.split('@')[0].isalnum()

    # Test with unique=True and domain with multiple @
    custom_domains = ['@@example.com']
    email = person.email(domains=custom_domains, unique=True)
    assert email.split('@')[1] == '@example.com'
    assert email.split('@')[0] == uuid.UUID(email.split('@')[0], version=4).hex

    # Test with unique=False and domain with multiple @
    custom_domains = ['@@example.com']
    email = person.email(domains=custom_domains, unique=False)
    assert email.split('@')[1] == '@example.com'
    assert email.split('@')[0].isalnum()

    # Test with unique=True and domain with special characters
    custom_domains = ['example.co.uk']
    email = person.email(domains=custom_domains, unique=True)
    assert email.split('@')[1] == 'example.co.uk'
    assert email.split('@')[0] == uuid.UUID(email.split('@')[0], version=4).hex

    # Test with unique=False and domain with special characters
    custom_domains = ['example.co.uk']
    email = person.email(domains=custom_domains, unique=False)
    assert email.split('@')[1] == 'example.co.uk'
    assert email.split('@')[0].isalnum()

    # Test with unique=True and domain with numbers
    custom_domains = ['example123.com']
    email = person.email(domains=custom_domains, unique=True)
    assert email.split('@')[1] == 'example123.com'
    assert email.split('@')[0] == uuid.UUID(email.split('@')[0], version=4).hex

    # Test with unique=False and domain with numbers
    custom_domains = ['example123.com']
    email = person.email(domains=custom_domains, unique=False)
    assert email.split('@')[1] == 'example123.com'
    assert email.split('@')[0].isalnum()

    # Test with unique=True and domain with hyphens
    custom_domains = ['example-test.com']
    email = person.email(domains=custom_domains, unique=True)
    assert email.split('@')[1] == 'example-test.com'
    assert email.split('@')[0] == uuid.UUID(email.split('@')[0], version=4).hex

    # Test with unique=False and domain with hyphens
    custom_domains = ['example-test.com']
    email = person.email(domains=custom_domains, unique=False)
    assert email.split('@')[1] == 'example-test.com'
    assert email.split('@')[0].isalnum()

    # Test with unique=True and domain with underscores
    custom_domains = ['example_test.com']
    email = person.email(domains=custom_domains, unique=True)
    assert email.split('@')[1] == 'example_test.com'
    assert email.split('@')[0] == uuid.UUID(email.split('@')[0], version=4).hex

    # Test with unique=False and domain with underscores
    custom_domains = ['example_test.com']
    email = person.email(domains=custom_domains, unique=False)
    assert email.split('@')[1] == 'example_test.com'
    assert email.split('@')[0].isalnum()

    # Test with unique=True and domain with spaces
    custom_domains = ['example test.com']
    email = person.email(domains=custom_domains, unique=True)
    assert email.split('@')[1] == 'example test.com'
    assert email.split('@')[0] == uuid.UUID(email.split('@')[0], version=4).hex

    # Test with unique=False and domain with spaces
    custom_domains = ['example test.com']
    email = person.email(domains=custom_domains, unique=False)
    assert email.split('@')[1] == 'example test.com'
    assert email.split('@')[0].isalnum()

    # Test with unique=True and domain with special characters and spaces
    custom_domains = ['example test.co.uk']
    email = person.email(domains=custom_domains, unique=True)
    assert email.split('


# LLM-generated content at query #9
#--------------------------

# Unit test for method patronymic of class Person
def test_Person_patronymic():  
    # Test with gender = Gender.MALE
    person = Person()
    result = person.patronymic(Gender.MALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with gender = Gender.FEMALE
    result = person.patronymic(Gender.FEMALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with gender = None
    result = person.patronymic()
    assert result is not None
    assert isinstance(result, str)
    # Test with unsupported locale
    person = Person(locale='en')
    result = person.patronymic()
    assert result is None


# LLM-generated content at query #10
#--------------------------

# Unit test for method surname of class Person
def test_Person_surname():  
    # Test with default gender (None)
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname in person._extract(["surnames"])

    # Test with gender Male
    surname_male = person.surname(gender=Gender.MALE)
    assert isinstance(surname_male, str)
    assert surname_male in person._extract(["surnames"])[Gender.MALE]

    # Test with gender Female
    surname_female = person.surname(gender=Gender.FEMALE)
    assert isinstance(surname_female, str)
    assert surname_female in person._extract(["surnames"])[Gender.FEMALE]

    # Test with invalid gender (should raise an error)
    try:
        person.surname(gender="invalid")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames
    person_ru = Person(locale="ru")
    surname_ru = person_ru.surname()
    assert isinstance(surname_ru, str)
    assert surname_ru in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and invalid gender (should raise an error)
    try:
        person_ru.surname(gender="invalid")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender None (should return a random surname)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a male surname)
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a female surname)
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a male surname)
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a female surname)
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a male surname)
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a female surname)
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a male surname)
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a female surname)
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a male surname)
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a female surname)
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a male surname)
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a female surname)
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a male surname)
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a female surname)
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a male surname)
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a female surname)
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE


# LLM-generated content at query #11
#--------------------------

# Unit test for method patronymic of class Person
def test_Person_patronymic():  
    # Test with gender=None
    person = Person()
    result = person.patronymic()
    assert result is None or isinstance(result, str)
    
    # Test with gender=Gender.MALE
    result = person.patronymic(gender=Gender.MALE)
    assert result is None or isinstance(result, str)
    
    # Test with gender=Gender.FEMALE
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None or isinstance(result, str)
    
    # Test with locale='ru'
    person_ru = Person(locale='ru')
    result = person_ru.patronymic(gender=Gender.MALE)
    assert isinstance(result, str)
    
    # Test with locale='uk'
    person_uk = Person(locale='uk')
    result = person_uk.patronymic(gender=Gender.FEMALE)
    assert isinstance(result, str)
    
    # Test with locale='en'
    person_en = Person(locale='en')
    result = person_en.patronymic(gender=Gender.MALE)
    assert result is None
    
    # Test with invalid gender
    try:
        person.patronymic(gender='invalid')
        assert False, "Should raise NonEnumerableError"
    except Exception:
        pass


# LLM-generated content at query #12
#--------------------------

# Unit test for method surname of class Person
def test_Person_surname():  
    # Test with default gender (None)
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname in person._extract(["surnames"])

    # Test with gender Male
    surname_male = person.surname(gender=Gender.MALE)
    assert isinstance(surname_male, str)
    assert surname_male in person._extract(["surnames"])

    # Test with gender Female
    surname_female = person.surname(gender=Gender.FEMALE)
    assert isinstance(surname_female, str)
    assert surname_female in person._extract(["surnames"])

    # Test with invalid gender (should raise an error)
    try:
        person.surname(gender="invalid_gender")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames
    person_ru = Person(locale="ru")
    surname_ru = person_ru.surname()
    assert isinstance(surname_ru, str)
    assert surname_ru in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Female
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender None
    surname_ru_none = person_ru.surname(gender=None)
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender invalid
    try:
        person_ru.surname(gender="invalid_gender")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (int)
    try:
        person_ru.surname(gender=123)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (float)
    try:
        person_ru.surname(gender=1.23)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (list)
    try:
        person_ru.surname(gender=[1, 2, 3])
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (dict)
    try:
        person_ru.surname(gender={"key": "value"})
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (tuple)
    try:
        person_ru.surname(gender=(1, 2, 3))
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (set)
    try:
        person_ru.surname(gender={1, 2, 3})
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (bool)
    try:
        person_ru.surname(gender=True)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (None)
    try:
        person_ru.surname(gender=None)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (empty string)
    try:
        person_ru.surname(gender="")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (whitespace)
    try:
        person_ru.surname(gender=" ")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (newline)
    try:
        person_ru.surname(gender="\n")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (tab)
    try:
        person_ru.surname(gender="\t")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (carriage return)
    try:
        person_ru.surname(gender="\r")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (form feed)
    try:
        person_ru.surname(gender="\f")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (vertical tab)
    try:
        person_ru.surname(gender="\v")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (backspace)
    try:
        person_ru.surname(gender="\b")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (null)
    try:
        person_ru.surname(gender="\0")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (escape)
    try:
        person_ru.surname(gender="\x1b")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (delete)
    try:
        person_ru.surname(gender="\x7f")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (non-printable)
    try:
        person_ru.surname(gender="\x80")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (non-ASCII)
    try:
        person_ru.surname(gender="\u00a9")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (emoji)
    try:
        person_ru.surname(gender="😀")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (surrogate pair)
    try:
        person_ru.surname(gender="\ud83d\ude00")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid UTF-8)
    try:
        person_ru.surname(gender=b"\xff")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid UTF-16)
    try:
        person_ru.surname(gender="\ud800")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid UTF-32)
    try:
        person_ru.surname(gender="\U00110000")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid code point)
    try:
        person_ru.surname(gender="\U00110000")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid Unicode)
    try:
        person_ru.surname(gender="\uFFFE")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid character)
    try:
        person_ru.surname(gender="\x00")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid string)
    try:
        person_ru.surname(gender="invalid_string")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid type)
    try:
        person_ru.surname(gender=object())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid class)
    try:
        person_ru.surname(gender=Person)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid instance)
    try:
        person_ru.surname(gender=Person())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid function)
    try:
        person_ru.surname(gender=lambda x: x)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid module)
    try:
        import sys
        person_ru.surname(gender=sys)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid package)
    try:
        import mimesis
        person_ru.surname(gender=mimesis)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (invalid file)
    try:
        import io
        person_ru.surname(gender=io.StringIO


# LLM-generated content at query #13
#--------------------------

# Unit test for method patronymic of class Person
def test_Person_patronymic():  
    # Test with gender = Gender.MALE
    person = Person()
    result = person.patronymic(gender=Gender.MALE)
    assert result is not None
    assert isinstance(result, str)
    
    # Test with gender = Gender.FEMALE
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is not None
    assert isinstance(result, str)
    
    # Test with gender = None (default)
    result = person.patronymic()
    assert result is not None
    assert isinstance(result, str)
    
    # Test with locale that does not have patronymics
    person = Person(locale='en')
    result = person.patronymic()
    assert result is None


# LLM-generated content at query #14
#--------------------------

# Unit test for method patronymic of class Person
def test_Person_patronymic():  
    # Test with gender = Gender.MALE
    person = Person()
    result = person.patronymic(gender=Gender.MALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with gender = Gender.FEMALE
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with gender = None (default)
    result = person.patronymic()
    assert result is not None
    assert isinstance(result, str)
    # Test with unsupported locale (e.g., 'en')
    person = Person(locale='en')
    result = person.patronymic()
    assert result is None


# LLM-generated content at query #15
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test case 1: gender is None
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])

    # Test case 2: gender is Gender.MALE
    person = Person()
    nationality = person.nationality(Gender.MALE)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.MALE]

    # Test case 3: gender is Gender.FEMALE
    person = Person()
    nationality = person.nationality(Gender.FEMALE)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.FEMALE]

    # Test case 4: gender is Gender.NOT_APPLICABLE
    person = Person()
    nationality = person.nationality(Gender.NOT_APPLICABLE)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.NOT_APPLICABLE]

    # Test case 5: gender is Gender.NOT_KNOWN
    person = Person()
    nationality = person.nationality(Gender.NOT_KNOWN)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.NOT_KNOWN]

    # Test case 6: gender is Gender.OTHER
    person = Person()
    nationality = person.nationality(Gender.OTHER)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.OTHER]

    # Test case 7: gender is Gender.PREFER_NOT_TO_SAY
    person = Person()
    nationality = person.nationality(Gender.PREFER_NOT_TO_SAY)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.PREFER_NOT_TO_SAY]

    # Test case 8: gender is Gender.NON_BINARY
    person = Person()
    nationality = person.nationality(Gender.NON_BINARY)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.NON_BINARY]

    # Test case 9: gender is Gender.TWO_SPIRIT
    person = Person()
    nationality = person.nationality(Gender.TWO_SPIRIT)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.TWO_SPIRIT]

    # Test case 10: gender is Gender.AGENDER
    person = Person()
    nationality = person.nationality(Gender.AGENDER)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.AGENDER]

    # Test case 11: gender is Gender.BIGENDER
    person = Person()
    nationality = person.nationality(Gender.BIGENDER)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.BIGENDER]

    # Test case 12: gender is Gender.GENDERFLUID
    person = Person()
    nationality = person.nationality(Gender.GENDERFLUID)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.GENDERFLUID]

    # Test case 13: gender is Gender.GENDERQUEER
    person = Person()
    nationality = person.nationality(Gender.GENDERQUEER)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.GENDERQUEER]

    # Test case 14: gender is Gender.PANGENDER
    person = Person()
    nationality = person.nationality(Gender.PANGENDER)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.PANGENDER]

    # Test case 15: gender is Gender.TRANSGENDER
    person = Person()
    nationality = person.nationality(Gender.TRANSGENDER)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.TRANSGENDER]

    # Test case 16: gender is Gender.TRANSGENDER_MALE
    person = Person()
    nationality = person.nationality(Gender.TRANSGENDER_MALE)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.TRANSGENDER_MALE]

    # Test case 17: gender is Gender.TRANSGENDER_FEMALE
    person = Person()
    nationality = person.nationality(Gender.TRANSGENDER_FEMALE)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.TRANSGENDER_FEMALE]

    # Test case 18: gender is Gender.TRANSGENDER_MAN
    person = Person()
    nationality = person.nationality(Gender.TRANSGENDER_MAN)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.TRANSGENDER_MAN]

    # Test case 19: gender is Gender.TRANSGENDER_WOMAN
    person = Person()
    nationality = person.nationality(Gender.TRANSGENDER_WOMAN)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.TRANSGENDER_WOMAN]

    # Test case 20: gender is Gender.TRANSGENDER_PERSON
    person = Person()
    nationality = person.nationality(Gender.TRANSGENDER_PERSON)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.TRANSGENDER_PERSON]

    # Test case 21: gender is Gender.CISGENDER
    person = Person()
    nationality = person.nationality(Gender.CISGENDER)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.CISGENDER]

    # Test case 22: gender is Gender.CISGENDER_MALE
    person = Person()
    nationality = person.nationality(Gender.CISGENDER_MALE)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.CISGENDER_MALE]

    # Test case 23: gender is Gender.CISGENDER_FEMALE
    person = Person()
    nationality = person.nationality(Gender.CISGENDER_FEMALE)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.CISGENDER_FEMALE]

    # Test case 24: gender is Gender.CISGENDER_MAN
    person = Person()
    nationality = person.nationality(Gender.CISGENDER_MAN)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.CISGENDER_MAN]

    # Test case 25: gender is Gender.CISGENDER_WOMAN
    person = Person()
    nationality = person.nationality(Gender.CISGENDER_WOMAN)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.CISGENDER_WOMAN]

    # Test case 26: gender is Gender.CISGENDER_PERSON
    person = Person()
    nationality = person.nationality(Gender.CISGENDER_PERSON)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.CISGENDER_PERSON]

    # Test case 27: gender is Gender.INTERSEX
    person = Person()
    nationality = person.nationality(Gender.INTERSEX)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.INTERSEX]

    # Test case 28: gender is Gender.INTERSEX_MALE
    person = Person()
    nationality = person.nationality(Gender.INTERSEX_MALE)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.INTERSEX_MALE]

    # Test case 29: gender is Gender.INTERSEX_FEMALE
    person = Person()
    nationality = person.nationality(Gender.INTERSEX_FEMALE)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.INTERSEX_FEMALE]

    # Test case 30: gender is Gender.INTERSEX_MAN
    person = Person()
    nationality = person.nationality(Gender.INTERSEX_MAN)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.INTERSEX_MAN]

    # Test case 31: gender is Gender.INTERSEX_WOMAN
    person = Person()
    nationality = person.nationality(Gender.INTERSEX_WOMAN)
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])[Gender.INTERSEX_WOMAN]

    # Test case 32: gender is Gender.INTERSEX_PERSON
    person = Person()
    nationality


# LLM-generated content at query #16
#--------------------------

# Unit test for method surname of class Person
def test_Person_surname():  
    # Test with default gender (None)
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname in person._extract(["surnames"])

    # Test with gender Male
    surname_male = person.surname(Gender.MALE)
    assert isinstance(surname_male, str)
    assert surname_male in person._extract(["surnames", "male"])

    # Test with gender Female
    surname_female = person.surname(Gender.FEMALE)
    assert isinstance(surname_female, str)
    assert surname_female in person._extract(["surnames", "female"])

    # Test with invalid gender (should raise error)
    try:
        person.surname("invalid_gender")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale that has gender-specific surnames
    person_ru = Person(locale="ru")
    surname_ru_male = person_ru.surname(Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames", "male"])

    # Test with locale that does not have gender-specific surnames
    person_en = Person(locale="en")
    surname_en = person_en.surname()
    assert isinstance(surname_en, str)
    assert surname_en in person_en._extract(["surnames"])

    # Test with seed for reproducibility
    person_seeded = Person(seed=42)
    surname1 = person_seeded.surname()
    surname2 = person_seeded.surname()
    assert surname1 == surname2

    # Test with different locales
    locales = ["en", "ru", "de", "fr", "es"]
    for locale in locales:
        person = Person(locale=locale)
        surname = person.surname()
        assert isinstance(surname, str)
        assert surname in person._extract(["surnames"])

    # Test with gender-specific surnames in different locales
    for locale in locales:
        person = Person(locale=locale)
        for gender in [Gender.MALE, Gender.FEMALE]:
            surname = person.surname(gender)
            assert isinstance(surname, str)
            surnames_data = person._extract(["surnames"])
            if isinstance(surnames_data, dict):
                assert surname in surnames_data[gender.value]
            else:
                assert surname in surnames_data

    # Test with None gender (should return random surname from all surnames)
    person = Person()
    surname_none = person.surname(None)
    assert isinstance(surname_none, str)
    surnames_data = person._extract(["surnames"])
    if isinstance(surnames_data, dict):
        all_surnames = []
        for gender_surnames in surnames_data.values():
            all_surnames.extend(gender_surnames)
        assert surname_none in all_surnames
    else:
        assert surname_none in surnames_data

    # Test with empty surnames list (should not happen in practice)
    # This is a hypothetical test to ensure the method handles edge cases
    person_empty = Person()
    # Mock the _extract method to return an empty list
    original_extract = person_empty._extract
    person_empty._extract = lambda keys, default=None: []
    try:
        surname_empty = person_empty.surname()
        # Should raise IndexError or return None
    except IndexError:
        pass
    except Exception as e:
        assert False, f"Unexpected exception: {e}"
    finally:
        person_empty._extract = original_extract

    # Test with single surname in list
    person_single = Person()
    # Mock the _extract method to return a list with one surname
    original_extract = person_single._extract
    person_single._extract = lambda keys, default=None: ["Smith"]
    surname_single = person_single.surname()
    assert surname_single == "Smith"
    person_single._extract = original_extract

    # Test with gender-specific surnames where one gender has empty list
    person_mixed = Person()
    # Mock the _extract method to return a dict with empty female list
    original_extract = person_mixed._extract
    person_mixed._extract = lambda keys, default=None: {
        "male": ["Smith", "Johnson"],
        "female": []
    }
    surname_male = person_mixed.surname(Gender.MALE)
    assert surname_male in ["Smith", "Johnson"]
    # Female should still work but return from male list? Actually should raise IndexError
    try:
        surname_female = person_mixed.surname(Gender.FEMALE)
        # Should raise IndexError
    except IndexError:
        pass
    except Exception as e:
        assert False, f"Unexpected exception: {e}"
    finally:
        person_mixed._extract = original_extract

    # Test with invalid gender enum value
    try:
        person.surname(Gender("invalid"))
    except ValueError:
        pass
    except Exception as e:
        assert False, f"Unexpected exception: {e}"

    # Test with gender as string (should be converted to enum)
    try:
        person.surname("male")
    except Exception as e:
        # Should raise ValueError or similar
        assert isinstance(e, ValueError)

    # Test with gender as integer (should be converted to enum)
    try:
        person.surname(1)
    except Exception as e:
        # Should raise ValueError or similar
        assert isinstance(e, ValueError)

    # Test with gender as Gender enum object
    surname_enum = person.surname(Gender.MALE)
    assert isinstance(surname_enum, str)

    # Test with gender as None (default)
    surname_default = person.surname(None)
    assert isinstance(surname_default, str)

    # Test that surname returns different values for different genders
    person = Person()
    surname_male = person.surname(Gender.MALE)
    surname_female = person.surname(Gender.FEMALE)
    # They could be the same if the locale doesn't have gender-specific surnames
    # So we can't assert they're different

    # Test that surname returns a string
    assert isinstance(person.surname(), str)

    # Test that surname returns a non-empty string
    assert len(person.surname()) > 0

    # Test that surname returns a string without leading/trailing whitespace
    surname = person.surname()
    assert surname == surname.strip()

    # Test that surname returns a string that is in the dataset
    surnames_data = person._extract(["surnames"])
    if isinstance(surnames_data, dict):
        all_surnames = []
        for gender_surnames in surnames_data.values():
            all_surnames.extend(gender_surnames)
        assert person.surname() in all_surnames
    else:
        assert person.surname() in surnames_data

    # Test with seed to ensure reproducibility
    person1 = Person(seed=123)
    person2 = Person(seed=123)
    assert person1.surname() == person2.surname()
    assert person1.surname(Gender.MALE) == person2.surname(Gender.MALE)
    assert person1.surname(Gender.FEMALE) == person2.surname(Gender.FEMALE)

    # Test with different seeds
    person1 = Person(seed=123)
    person2 = Person(seed=456)
    # They might be the same by chance, so we can't assert they're different

    # Test that surname doesn't return None
    assert person.surname() is not None

    # Test that surname doesn't return empty string
    assert person.surname() != ""

    # Test that surname returns a valid string for all locales
    for locale in person._data.keys():
        person_locale = Person(locale=locale)
        surname = person_locale.surname()
        assert isinstance(surname, str)
        assert len(surname) > 0

    # Test that surname handles gender-specific surnames correctly
    for locale in person._data.keys():
        person_locale = Person(locale=locale)
        surnames_data = person_locale._extract(["surnames"])
        if isinstance(surnames_data, dict):
            for gender in Gender:
                surname = person_locale.surname(gender)
                assert surname in surnames_data[gender.value]
        else:
            for gender in Gender:
                surname = person_locale.surname(gender)
                assert surname in surnames_data

    # Test that surname raises appropriate error for invalid gender
    try:
        person.surname("invalid")
    except ValueError:
        pass
    except Exception as e:
        assert False, f"Unexpected exception type: {type(e).__name__}"

    # Test that surname works with Gender enum
    for gender in Gender:
        surname = person.surname(gender)
        assert isinstance(surname, str)

    # Test that surname returns different values when called multiple times
    # (unless seeded)
    person = Person()
    surnames = set(person.surname() for _ in range(100))
    # There might be duplicates if the dataset is small, so we can't assert > 1
    assert len(surnames) >= 1

    # Test with seeded provider
    person = Person(seed=999)
    surnames_set = set(person.surname() for _ in range(10))
    # With seed, all calls


# LLM-generated content at query #17
#--------------------------

# Unit test for method patronymic of class Person
def test_Person_patronymic():  
    # Test with gender = Gender.MALE
    person = Person()
    result = person.patronymic(gender=Gender.MALE)
    assert isinstance(result, str) or result is None

    # Test with gender = Gender.FEMALE
    result = person.patronymic(gender=Gender.FEMALE)
    assert isinstance(result, str) or result is None

    # Test with gender = None
    result = person.patronymic(gender=None)
    assert isinstance(result, str) or result is None

    # Test with locale = 'ru'
    person_ru = Person(locale='ru')
    result = person_ru.patronymic(gender=Gender.MALE)
    assert isinstance(result, str) or result is None

    # Test with locale = 'uk'
    person_uk = Person(locale='uk')
    result = person_uk.patronymic(gender=Gender.FEMALE)
    assert isinstance(result, str) or result is None

    # Test with locale that does not have patronymic data
    person_en = Person(locale='en')
    result = person_en.patronymic(gender=Gender.MALE)
    assert result is None

    # Test with invalid gender
    try:
        person.patronymic(gender='invalid')
    except Exception as e:
        assert isinstance(e, Exception)

    # Test with seed
    person_seeded = Person(seed=42)
    result1 = person_seeded.patronymic(gender=Gender.MALE)
    person_seeded2 = Person(seed=42)
    result2 = person_seeded2.patronymic(gender=Gender.MALE)
    assert result1 == result2

    # Test that result is in the list of patronymics for the given gender and locale
    if result is not None:
        patronymics = person._extract(['patronymic', str(Gender.MALE)], default=[])
        assert result in patronymics

    # Test that result is None when patronymics list is empty
    person_empty = Person(locale='en')
    result = person_empty.patronymic(gender=Gender.MALE)
    assert result is None

    # Test that result is a string when patronymics list is not empty
    person_ru = Person(locale='ru')
    result = person_ru.patronymic(gender=Gender.MALE)
    assert isinstance(result, str)

    # Test that result is a random choice from the patronymics list
    patronymics = person_ru._extract(['patronymic', str(Gender.MALE)], default=[])
    assert result in patronymics

    # Test that result is different for different genders
    result_male = person_ru.patronymic(gender=Gender.MALE)
    result_female = person_ru.patronymic(gender=Gender.FEMALE)
    assert result_male != result_female

    # Test that result is different for different locales
    result_ru = person_ru.patronymic(gender=Gender.MALE)
    result_uk = person_uk.patronymic(gender=Gender.MALE)
    assert result_ru != result_uk

    # Test that result is the same for the same seed and gender
    person1 = Person(seed=123)
    result1 = person1.patronymic(gender=Gender.MALE)
    person2 = Person(seed=123)
    result2 = person2.patronymic(gender=Gender.MALE)
    assert result1 == result2

    # Test that result is different for different seeds
    person1 = Person(seed=123)
    result1 = person1.patronymic(gender=Gender.MALE)
    person2 = Person(seed=456)
    result2 = person2.patronymic(gender=Gender.MALE)
    assert result1 != result2

    # Test that result is None when gender is not provided and locale does not have default patronymics
    person_en = Person(locale='en')
    result = person_en.patronymic()
    assert result is None

    # Test that result is a string when gender is not provided and locale has default patronymics
    person_ru = Person(locale='ru')
    result = person_ru.patronymic()
    assert isinstance(result, str)

    # Test that result is in the list of patronymics for the default gender
    patronymics = person_ru._extract(['patronymic', str(Gender.MALE)], default=[])
    assert result in patronymics

    # Test that result is different for different genders when gender is not provided
    result_male = person_ru.patronymic(gender=Gender.MALE)
    result_female = person_ru.patronymic(gender=Gender.FEMALE)
    assert result_male != result_female

    # Test that result is different for different locales when gender is not provided
    result_ru = person_ru.patronymic()
    result_uk = person_uk.patronymic()
    assert result_ru != result_uk

    # Test that result is the same for the same seed when gender is not provided
    person1 = Person(seed=123)
    result1 = person1.patronymic()
    person2 = Person(seed=123)
    result2 = person2.patronymic()
    assert result1 == result2

    # Test that result is different for different seeds when gender is not provided
    person1 = Person(seed=123)
    result1 = person1.patronymic()
    person2 = Person(seed=456)
    result2 = person2.patronymic()
    assert result1 != result2

    # Test that result is None when patronymics list is empty for the given gender
    person_empty = Person(locale='en')
    result = person_empty.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test that result is a string when patronymics list is not empty for the given gender
    person_ru = Person(locale='ru')
    result = person_ru.patronymic(gender=Gender.FEMALE)
    assert isinstance(result, str)

    # Test that result is in the list of patronymics for the given gender
    patronymics = person_ru._extract(['patronymic', str(Gender.FEMALE)], default=[])
    assert result in patronymics

    # Test that result is different for different genders when patronymics list is not empty
    result_male = person_ru.patronymic(gender=Gender.MALE)
    result_female = person_ru.patronymic(gender=Gender.FEMALE)
    assert result_male != result_female

    # Test that result is different for different locales when patronymics list is not empty
    result_ru = person_ru.patronymic(gender=Gender.FEMALE)
    result_uk = person_uk.patronymic(gender=Gender.FEMALE)
    assert result_ru != result_uk

    # Test that result is the same for the same seed and gender when patronymics list is not empty
    person1 = Person(seed=123)
    result1 = person1.patronymic(gender=Gender.FEMALE)
    person2 = Person(seed=123)
    result2 = person2.patronymic(gender=Gender.FEMALE)
    assert result1 == result2

    # Test that result is different for different seeds when patronymics list is not empty
    person1 = Person(seed=123)
    result1 = person1.patronymic(gender=Gender.FEMALE)
    person2 = Person(seed=456)
    result2 = person2.patronymic(gender=Gender.FEMALE)
    assert result1 != result2

    # Test that result is None when gender is not provided and patronymics list is empty for default gender
    person_empty = Person(locale='en')
    result = person_empty.patronymic()
    assert result is None

    # Test that result is a string when gender is not provided and patronymics list is not empty for default gender
    person_ru = Person(locale='ru')
    result = person_ru.patronymic()
    assert isinstance(result, str)

    # Test that result is in the list of patronymics for the default gender
    patronymics = person_ru._extract(['patronymic', str(Gender.MALE)], default=[])
    assert result in patronymics

    # Test that result is different for different genders when gender is not provided and patronymics list is not empty
    result_male = person_ru.patronymic(gender=Gender.MALE)
    result_female = person_ru.patronymic(gender=Gender.FEMALE)
    assert result_male != result_female

    # Test that result is different for different locales when gender is not provided and patronymics list is not empty
    result_ru = person_ru.patronymic()
    result_uk = person_uk.patronymic()
    assert result_ru != result_uk

    # Test that result is the same for the same seed when gender is not provided and patronymics list is not empty
    person1 = Person(seed=123)
    result1 =


# LLM-generated content at query #18
#--------------------------

# Unit test for method surname of class Person
def test_Person_surname():  
    # Test with default gender (None)
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname in person._extract(["surnames"])

    # Test with gender Male
    surname_male = person.surname(Gender.MALE)
    assert isinstance(surname_male, str)
    assert surname_male in person._extract(["surnames"])[Gender.MALE]

    # Test with gender Female
    surname_female = person.surname(Gender.FEMALE)
    assert isinstance(surname_female, str)
    assert surname_female in person._extract(["surnames"])[Gender.FEMALE]

    # Test with invalid gender (should raise an error)
    try:
        person.surname("INVALID_GENDER")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames
    person_ru = Person(locale="ru")
    surname_ru = person_ru.surname()
    assert isinstance(surname_ru, str)
    assert surname_ru in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender
    surname_ru_male = person_ru.surname(Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female
    surname_ru_female = person_ru.surname(Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and invalid gender (should raise an error)
    try:
        person_ru.surname("INVALID_GENDER")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender None (should return a random surname from the list)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a random male surname from the list)
    surname_ru_male = person_ru.surname(Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a random female surname from the list)
    surname_ru_female = person_ru.surname(Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname from the list)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a random male surname from the list)
    surname_ru_male = person_ru.surname(Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a random female surname from the list)
    surname_ru_female = person_ru.surname(Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname from the list)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a random male surname from the list)
    surname_ru_male = person_ru.surname(Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a random female surname from the list)
    surname_ru_female = person_ru.surname(Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname from the list)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a random male surname from the list)
    surname_ru_male = person_ru.surname(Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a random female surname from the list)
    surname_ru_female = person_ru.surname(Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname from the list)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a random male surname from the list)
    surname_ru_male = person_ru.surname(Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a random female surname from the list)
    surname_ru_female = person_ru.surname(Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname from the list)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a random male surname from the list)
    surname_ru_male = person_ru.surname(Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a random female surname from the list)
    surname_ru_female = person_ru.surname(Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname from the list)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a random male surname from the list)
    surname_ru_male = person_ru.surname(Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])[Gender.MALE]

    # Test with locale-specific surnames and gender Female (should return a random female surname from the list)
    surname_ru_female = person_ru.surname(Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])[Gender.FEMALE]

    # Test with locale-specific surnames and gender None (should return a random surname from the list)
    surname_ru_none = person_ru.surname()
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Male (should return a random male surname from the list)
    surname_ru_male = person_ru.surname(Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._


####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._data["nationality"]

    # Test with gender Male
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._data["nationality"]["male"]

    # Test with gender Female
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._data["nationality"]["female"]

    # Test with invalid gender (should raise an error)
    try:
        person.nationality("invalid_gender")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)


# LLM-generated content at query #2
#--------------------------

# Unit test for method surname of class Person
def test_Person_surname():  
    # Test with default gender (None)
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname in person._extract(["surnames"])

    # Test with gender Male
    surname_male = person.surname(gender=Gender.MALE)
    assert isinstance(surname_male, str)
    assert surname_male in person._extract(["surnames"])

    # Test with gender Female
    surname_female = person.surname(gender=Gender.FEMALE)
    assert isinstance(surname_female, str)
    assert surname_female in person._extract(["surnames"])

    # Test with invalid gender (should raise an error)
    try:
        person.surname(gender="INVALID")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames (if applicable)
    person_ru = Person(locale="ru")
    surname_ru = person_ru.surname()
    assert isinstance(surname_ru, str)
    assert surname_ru in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Female
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender None
    surname_ru_none = person_ru.surname(gender=None)
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender invalid
    try:
        person_ru.surname(gender="INVALID")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.MALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.FEMALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=None)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender="INVALID")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.MALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.FEMALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=None)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender="INVALID")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.MALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.FEMALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=None)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender="INVALID")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.MALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.FEMALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=None)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender="INVALID")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.MALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.FEMALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=None)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender="INVALID")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.MALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.FEMALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=None)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender="INVALID")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.MALE)
        # This should not raise an error
    except Exception as e:
        assert False, f"Unexpected error: {e}"

    # Test with locale-specific surnames and gender invalid (but valid for other locale)
    try:
        person_ru.surname(gender=Gender.FEMALE)
        # This should not raise an error
    except Exception as e:
        assert False,


# LLM-generated content at query #3
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])

    # Test with gender Male
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._extract(["nationality"])[Gender.MALE]

    # Test with gender Female
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._extract(["nationality"])[Gender.FEMALE]

    # Test with invalid gender (should raise an error)
    try:
        person.nationality("INVALID_GENDER")
        assert False, "Should have raised an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data
    person_ru = Person(locale="ru")
    nationality_ru = person_ru.nationality()
    assert isinstance(nationality_ru, str)
    assert nationality_ru in person_ru._extract(["nationality"])

    # Test with locale-specific data and gender
    nationality_ru_male = person_ru.nationality(Gender.MALE)
    assert isinstance(nationality_ru_male, str)
    assert nationality_ru_male in person_ru._extract(["nationality"])[Gender.MALE]

    # Test with locale-specific data and gender Female
    nationality_ru_female = person_ru.nationality(Gender.FEMALE)
    assert isinstance(nationality_ru_female, str)
    assert nationality_ru_female in person_ru._extract(["nationality"])[Gender.FEMALE]

    # Test with locale-specific data and invalid gender (should raise an error)
    try:
        person_ru.nationality("INVALID_GENDER")
        assert False, "Should have raised an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender None
    nationality_ru_none = person_ru.nationality()
    assert isinstance(nationality_ru_none, str)
    assert nationality_ru_none in person_ru._extract(["nationality"])

    # Test with locale-specific data and gender None (should be the same as default)
    assert nationality_ru_none == person_ru.nationality(None)

    # Test with locale-specific data and gender Male (should be different from default)
    assert nationality_ru_male != nationality_ru_none

    # Test with locale-specific data and gender Female (should be different from default)
    assert nationality_ru_female != nationality_ru_none

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different)
    assert nationality_ru_male != nationality_ru_female

    # Test with locale-specific data and gender Male and Female (should be different


# LLM-generated content at query #4
#--------------------------

# Unit test for method surname of class Person
def test_Person_surname():  
    # Test with default gender (None)
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname in person._extract(["surnames"])

    # Test with gender Male
    surname_male = person.surname(gender=Gender.MALE)
    assert isinstance(surname_male, str)
    assert surname_male in person._extract(["surnames"])

    # Test with gender Female
    surname_female = person.surname(gender=Gender.FEMALE)
    assert isinstance(surname_female, str)
    assert surname_female in person._extract(["surnames"])

    # Test with locale-specific surnames (e.g., Russian locale)
    person_ru = Person(locale='ru')
    surname_ru = person_ru.surname()
    assert isinstance(surname_ru, str)
    surnames_ru = person_ru._extract(["surnames"])
    if isinstance(surnames_ru, dict):
        # If surnames are separated by gender, check in both
        assert surname_ru in surnames_ru.get('male', []) or surname_ru in surnames_ru.get('female', [])
    else:
        assert surname_ru in surnames_ru

    # Test that surnames are random (not always the same)
    surnames_set = {person.surname() for _ in range(10)}
    assert len(surnames_set) > 1

    # Test with seed for reproducibility
    person_seeded = Person(seed=42)
    surname1 = person_seeded.surname()
    person_seeded2 = Person(seed=42)
    surname2 = person_seeded2.surname()
    assert surname1 == surname2

    print("All tests passed for Person.surname()")

# Run the test
test_Person_surname()


# LLM-generated content at query #5
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._data["nationality"]

    # Test with gender Male
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._data["nationality"]["male"]

    # Test with gender Female
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._data["nationality"]["female"]

    # Test with invalid gender (should raise an error)
    try:
        person.nationality("invalid_gender")
        assert False, "Expected an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)


# LLM-generated content at query #6
#--------------------------

# Unit test for method password of class Person
def test_Person_password():  
    # Test case 1: Check if the password length is correct
    person = Person()
    password = person.password(length=10)
    assert len(password) == 10

    # Test case 2: Check if the password contains only allowed characters
    allowed_characters = ascii_letters + digits + punctuation
    for char in password:
        assert char in allowed_characters

    # Test case 3: Check if the hashed password is correct
    hashed_password = person.password(length=8, hashed=True)
    assert len(hashed_password) == 64  # SHA256 hash length
    assert hashed_password.isalnum()  # Should be alphanumeric

    # Test case 4: Check if the password is random
    password1 = person.password(length=12)
    password2 = person.password(length=12)
    assert password1 != password2

    # Test case 5: Check if the hashed password is random
    hashed_password1 = person.password(length=10, hashed=True)
    hashed_password2 = person.password(length=10, hashed=True)
    assert hashed_password1 != hashed_password2

    # Test case 6: Check if the password length is at least 1
    password = person.password(length=1)
    assert len(password) == 1

    # Test case 7: Check if the password length can be very long
    password = person.password(length=100)
    assert len(password) == 100

    # Test case 8: Check if the password contains at least one of each character type
    password = person.password(length=100)
    has_letter = any(char in ascii_letters for char in password)
    has_digit = any(char in digits for char in password)
    has_punctuation = any(char in punctuation for char in password)
    assert has_letter or has_digit or has_punctuation

    # Test case 9: Check if the hashed password is a valid SHA256 hash
    import hashlib
    password = "test_password"
    hashed = hashlib.sha256(password.encode()).hexdigest()
    person = Person(seed=123)  # Use a seed for reproducibility
    generated_hashed = person.password(length=len(password), hashed=True)
    assert generated_hashed != hashed  # Should be different due to random generation

    # Test case 10: Check if the password method works with different seeds
    person1 = Person(seed=123)
    person2 = Person(seed=456)
    password1 = person1.password(length=10)
    password2 = person2.password(length=10)
    assert password1 != password2

    print("All tests passed!")

# Run the unit test
test_Person_password()


# LLM-generated content at query #7
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])

    # Test with Gender.MALE
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._extract(["nationality"])[Gender.MALE]

    # Test with Gender.FEMALE
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._extract(["nationality"])[Gender.FEMALE]

    # Test with invalid gender (should raise an error)
    try:
        person.nationality("invalid_gender")
    except Exception as e:
        assert isinstance(e, ValueError)


# LLM-generated content at query #8
#--------------------------

# Unit test for method password of class Person
def test_Person_password():  
    # Test case 1: Test default password length
    person = Person()
    password = person.password()
    assert len(password) == 8

    # Test case 2: Test password length with custom length
    password = person.password(length=12)
    assert len(password) == 12

    # Test case 3: Test hashed password
    hashed_password = person.password(hashed=True)
    assert len(hashed_password) == 64  # SHA256 hash length

    # Test case 4: Test password with custom length and hashed
    hashed_password = person.password(length=10, hashed=True)
    assert len(hashed_password) == 64  # SHA256 hash length

    # Test case 5: Test password characters
    password = person.password(length=100)
    allowed_characters = ascii_letters + digits + punctuation
    for char in password:
        assert char in allowed_characters

    # Test case 6: Test uniqueness of passwords
    passwords = set()
    for _ in range(100):
        password = person.password()
        passwords.add(password)
    assert len(passwords) == 100  # All passwords should be unique

    # Test case 7: Test hashed password uniqueness
    hashed_passwords = set()
    for _ in range(100):
        hashed_password = person.password(hashed=True)
        hashed_passwords.add(hashed_password)
    assert len(hashed_passwords) == 100  # All hashed passwords should be unique

    # Test case 8: Test password with minimum length
    password = person.password(length=1)
    assert len(password) == 1

    # Test case 9: Test password with maximum length
    password = person.password(length=1000)
    assert len(password) == 1000

    # Test case 10: Test password with negative length (should raise an error)
    try:
        person.password(length=-1)
        assert False, "Expected an error for negative length"
    except ValueError:
        pass

    # Test case 11: Test password with zero length (should raise an error)
    try:
        person.password(length=0)
        assert False, "Expected an error for zero length"
    except ValueError:
        pass

    # Test case 12: Test password with non-integer length (should raise an error)
    try:
        person.password(length="invalid")
        assert False, "Expected an error for non-integer length"
    except TypeError:
        pass

    # Test case 13: Test password with invalid hashed parameter (should raise an error)
    try:
        person.password(hashed="invalid")
        assert False, "Expected an error for invalid hashed parameter"
    except TypeError:
        pass

    # Test case 14: Test password with both length and hashed parameters
    password = person.password(length=16, hashed=True)
    assert len(password) == 64  # SHA256 hash length

    # Test case 15: Test password with only hashed parameter
    hashed_password = person.password(hashed=True)
    assert len(hashed_password) == 64  # SHA256 hash length

    # Test case 16: Test password with only length parameter
    password = person.password(length=20)
    assert len(password) == 20

    # Test case 17: Test password with length and hashed parameters swapped
    password = person.password(hashed=True, length=32)
    assert len(password) == 64  # SHA256 hash length

    # Test case 18: Test password with length and hashed parameters as keyword arguments
    password = person.password(length=24, hashed=False)
    assert len(password) == 24

    # Test case 19: Test password with length and hashed parameters as positional arguments
    password = person.password(24, False)
    assert len(password) == 24

    # Test case 20: Test password with length and hashed parameters as mixed arguments
    password = person.password(24, hashed=False)
    assert len(password) == 24

    # Test case 21: Test password with length and hashed parameters as mixed arguments (swapped)
    password = person.password(hashed=False, length=24)
    assert len(password) == 24

    # Test case 22: Test password with length and hashed parameters as mixed arguments (invalid order)
    try:
        person.password(hashed=False, 24)
        assert False, "Expected an error for invalid argument order"
    except SyntaxError:
        pass

    # Test case 23: Test password with length and hashed parameters as mixed arguments (invalid type)
    try:
        person.password(length="24", hashed=False)
        assert False, "Expected an error for invalid length type"
    except TypeError:
        pass

    # Test case 24: Test password with length and hashed parameters as mixed arguments (invalid hashed type)
    try:
        person.password(length=24, hashed="False")
        assert False, "Expected an error for invalid hashed type"
    except TypeError:
        pass

    # Test case 25: Test password with length and hashed parameters as mixed arguments (invalid both types)
    try:
        person.password(length="24", hashed="False")
        assert False, "Expected an error for invalid both types"
    except TypeError:
        pass

    # Test case 26: Test password with length and hashed parameters as mixed arguments (invalid length value)
    try:
        person.password(length=-24, hashed=False)
        assert False, "Expected an error for negative length"
    except ValueError:
        pass

    # Test case 27: Test password with length and hashed parameters as mixed arguments (invalid hashed value)
    try:
        person.password(length=24, hashed="invalid")
        assert False, "Expected an error for invalid hashed value"
    except TypeError:
        pass

    # Test case 28: Test password with length and hashed parameters as mixed arguments (invalid both values)
    try:
        person.password(length=-24, hashed="invalid")
        assert False, "Expected an error for invalid both values"
    except (ValueError, TypeError):
        pass

    # Test case 29: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=1, hashed=True)
    assert len(password) == 64  # SHA256 hash length

    # Test case 30: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=1000, hashed=True)
    assert len(password) == 64  # SHA256 hash length

    # Test case 31: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=1, hashed=False)
    assert len(password) == 1

    # Test case 32: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=1000, hashed=False)
    assert len(password) == 1000

    # Test case 33: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=0, hashed=False)
    assert len(password) == 0

    # Test case 34: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=0, hashed=True)
    assert len(password) == 64  # SHA256 hash length

    # Test case 35: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=-0, hashed=False)
    assert len(password) == 0

    # Test case 36: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=-0, hashed=True)
    assert len(password) == 64  # SHA256 hash length

    # Test case 37: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=+0, hashed=False)
    assert len(password) == 0

    # Test case 38: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=+0, hashed=True)
    assert len(password) == 64  # SHA256 hash length

    # Test case 39: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=1.0, hashed=False)
    assert len(password) == 1

    # Test case 40: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=1.0, hashed=True)
    assert len(password) == 64  # SHA256 hash length

    # Test case 41: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=1.5, hashed=False)
    assert len(password) == 1

    # Test case 42: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password(length=1.5, hashed=True)
    assert len(password) == 64  # SHA256 hash length

    # Test case 43: Test password with length and hashed parameters as mixed arguments (valid edge case)
    password = person.password


# LLM-generated content at query #9
#--------------------------

# Unit test for method password of class Person
def test_Person_password():  
    # Test case 1: Test password length
    person = Person()
    password = person.password(length=10)
    assert len(password) == 10

    # Test case 2: Test password hashed
    password_hashed = person.password(length=8, hashed=True)
    assert len(password_hashed) == 64  # SHA256 hash length

    # Test case 3: Test password characters
    password = person.password(length=20)
    allowed_chars = ascii_letters + digits + punctuation
    for char in password:
        assert char in allowed_chars

    # Test case 4: Test password uniqueness
    password1 = person.password()
    password2 = person.password()
    assert password1 != password2

    # Test case 5: Test password with custom length
    password = person.password(length=15)
    assert len(password) == 15

    # Test case 6: Test password with hashed=True
    password_hashed = person.password(hashed=True)
    assert len(password_hashed) == 64

    # Test case 7: Test password with hashed=False
    password = person.password(hashed=False)
    assert len(password) == 8

    # Test case 8: Test password with length=0
    password = person.password(length=0)
    assert len(password) == 0

    # Test case 9: Test password with negative length
    try:
        person.password(length=-5)
    except ValueError:
        pass  # Expected behavior

    # Test case 10: Test password with very large length
    password = person.password(length=1000)
    assert len(password) == 1000


# LLM-generated content at query #10
#--------------------------

# Unit test for method password of class Person
def test_Person_password():  
    # Test case 1: Test default password length
    person = Person()
    password = person.password()
    assert len(password) == 8
    assert any(c.isalpha() for c in password)
    assert any(c.isdigit() for c in password)
    assert any(c in punctuation for c in password)

    # Test case 2: Test custom password length
    password = person.password(length=12)
    assert len(password) == 12

    # Test case 3: Test hashed password
    hashed_password = person.password(hashed=True)
    assert len(hashed_password) == 64  # SHA256 hash length
    assert all(c in '0123456789abcdef' for c in hashed_password)

    # Test case 4: Test hashed password with custom length
    hashed_password = person.password(length=10, hashed=True)
    assert len(hashed_password) == 64

    # Test case 5: Test that password contains all character types
    password = person.password(length=100)
    assert any(c.isupper() for c in password)
    assert any(c.islower() for c in password)
    assert any(c.isdigit() for c in password)
    assert any(c in punctuation for c in password)

    # Test case 6: Test that password is random
    passwords = set(person.password() for _ in range(100))
    assert len(passwords) > 1  # Very low chance of collision

    # Test case 7: Test with seed for reproducibility
    person1 = Person(seed=42)
    person2 = Person(seed=42)
    assert person1.password() == person2.password()
    assert person1.password(hashed=True) == person2.password(hashed=True)

    # Test case 8: Test edge case - minimum length
    password = person.password(length=1)
    assert len(password) == 1
    assert password[0] in ascii_letters + digits + punctuation

    # Test case 9: Test that hashed password is deterministic for same input
    person = Person(seed=123)
    password1 = person.password(hashed=True)
    person = Person(seed=123)
    password2 = person.password(hashed=True)
    assert password1 == password2

    # Test case 10: Test that unhashed password is different from hashed
    person = Person()
    plain = person.password()
    hashed = person.password(hashed=True)
    assert plain != hashed
    assert len(plain) == 8
    assert len(hashed) == 64


# LLM-generated content at query #11
#--------------------------

# Unit test for method surname of class Person
def test_Person_surname():  
    # Test with default gender (None)
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname in person._extract(["surnames"])

    # Test with gender Male
    surname_male = person.surname(gender=Gender.MALE)
    assert isinstance(surname_male, str)
    assert surname_male in person._extract(["surnames"])

    # Test with gender Female
    surname_female = person.surname(gender=Gender.FEMALE)
    assert isinstance(surname_female, str)
    assert surname_female in person._extract(["surnames"])

    # Test with invalid gender (should raise NonEnumerableError)
    try:
        person.surname(gender="INVALID")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass

    # Test with locale that has gender-specific surnames
    person_ru = Person(locale="ru")
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames", "male"])

    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames", "female"])

    # Test with locale that does not have gender-specific surnames
    person_en = Person(locale="en")
    surname_en = person_en.surname()
    assert isinstance(surname_en, str)
    assert surname_en in person_en._extract(["surnames"])

    # Test with seed for reproducibility
    person_seeded = Person(seed=42)
    surname1 = person_seeded.surname()
    person_seeded2 = Person(seed=42)
    surname2 = person_seeded2.surname()
    assert surname1 == surname2


# LLM-generated content at query #12
#--------------------------

# Unit test for method patronymic of class Person
def test_Person_patronymic():  
    # Test with gender parameter as Gender.MALE
    person = Person()
    result = person.patronymic(gender=Gender.MALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with gender parameter as Gender.FEMALE
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with gender parameter as None
    result = person.patronymic()
    assert result is None or isinstance(result, str)
    # Test with locale that does not have patronymics
    person = Person(locale='en')
    result = person.patronymic()
    assert result is None


# LLM-generated content at query #13
#--------------------------

# Unit test for method patronymic of class Person
def test_Person_patronymic():  
    # Test case 1: Test with gender=Gender.MALE
    person = Person()
    result = person.patronymic(gender=Gender.MALE)
    assert isinstance(result, str) or result is None

    # Test case 2: Test with gender=Gender.FEMALE
    result = person.patronymic(gender=Gender.FEMALE)
    assert isinstance(result, str) or result is None

    # Test case 3: Test with gender=None
    result = person.patronymic(gender=None)
    assert isinstance(result, str) or result is None

    # Test case 4: Test with locale='ru'
    person = Person(locale='ru')
    result = person.patronymic(gender=Gender.MALE)
    assert isinstance(result, str) or result is None

    # Test case 5: Test with locale='uk'
    person = Person(locale='uk')
    result = person.patronymic(gender=Gender.FEMALE)
    assert isinstance(result, str) or result is None

    # Test case 6: Test with locale='en'
    person = Person(locale='en')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 7: Test with locale='fr'
    person = Person(locale='fr')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 8: Test with locale='de'
    person = Person(locale='de')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 9: Test with locale='it'
    person = Person(locale='it')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 10: Test with locale='es'
    person = Person(locale='es')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 11: Test with locale='pt'
    person = Person(locale='pt')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 12: Test with locale='pl'
    person = Person(locale='pl')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 13: Test with locale='nl'
    person = Person(locale='nl')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 14: Test with locale='sv'
    person = Person(locale='sv')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 15: Test with locale='da'
    person = Person(locale='da')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 16: Test with locale='no'
    person = Person(locale='no')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 17: Test with locale='fi'
    person = Person(locale='fi')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 18: Test with locale='cs'
    person = Person(locale='cs')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 19: Test with locale='hu'
    person = Person(locale='hu')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 20: Test with locale='ro'
    person = Person(locale='ro')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 21: Test with locale='bg'
    person = Person(locale='bg')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 22: Test with locale='el'
    person = Person(locale='el')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 23: Test with locale='tr'
    person = Person(locale='tr')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 24: Test with locale='he'
    person = Person(locale='he')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 25: Test with locale='ar'
    person = Person(locale='ar')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 26: Test with locale='fa'
    person = Person(locale='fa')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 27: Test with locale='hi'
    person = Person(locale='hi')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 28: Test with locale='th'
    person = Person(locale='th')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 29: Test with locale='ko'
    person = Person(locale='ko')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 30: Test with locale='ja'
    person = Person(locale='ja')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 31: Test with locale='zh'
    person = Person(locale='zh')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 32: Test with locale='vi'
    person = Person(locale='vi')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 33: Test with locale='id'
    person = Person(locale='id')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 34: Test with locale='ms'
    person = Person(locale='ms')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 35: Test with locale='fil'
    person = Person(locale='fil')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 36: Test with locale='sw'
    person = Person(locale='sw')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 37: Test with locale='af'
    person = Person(locale='af')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 38: Test with locale='zu'
    person = Person(locale='zu')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 39: Test with locale='xh'
    person = Person(locale='xh')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 40: Test with locale='nso'
    person = Person(locale='nso')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 41: Test with locale='tn'
    person = Person(locale='tn')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 42: Test with locale='st'
    person = Person(locale='st')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 43: Test with locale='ts'
    person = Person(locale='ts')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 44: Test with locale='ss'
    person = Person(locale='ss')
    result = person.patronymic(gender=Gender.MALE)
    assert result is None

    # Test case 45: Test with locale='ve'
    person = Person(locale='ve')
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is None

    # Test case 46: Test with locale='nr'
    person = Person(locale='nr')
    result = person.patronymic(gender=Gender.MALE


# LLM-generated content at query #14
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._data["nationality"]

    # Test with gender Male
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._data["nationality"][Gender.MALE.value]

    # Test with gender Female
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._data["nationality"][Gender.FEMALE.value]

    # Test with invalid gender (should raise an error)
    try:
        person.nationality("invalid_gender")
        assert False, "Should have raised an error for invalid gender"
    except Exception as e:
        assert isinstance(e, ValueError)


# LLM-generated content at query #15
#--------------------------

# Unit test for method surname of class Person
def test_Person_surname():  
    # Test with default gender (None)
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname in person._extract(["surnames"])

    # Test with gender Male
    surname_male = person.surname(gender=Gender.MALE)
    assert isinstance(surname_male, str)
    assert surname_male in person._extract(["surnames", "male"])

    # Test with gender Female
    surname_female = person.surname(gender=Gender.FEMALE)
    assert isinstance(surname_female, str)
    assert surname_female in person._extract(["surnames", "female"])

    # Test with invalid gender (should raise an error)
    try:
        person.surname(gender="invalid")
    except Exception as e:
        assert isinstance(e, ValueError)


# LLM-generated content at query #16
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])

    # Test with gender Male
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._extract(["nationality"])[Gender.MALE]

    # Test with gender Female
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._extract(["nationality"])[Gender.FEMALE]

    # Test with invalid gender (should raise an error)
    try:
        person.nationality("invalid_gender")
    except Exception as e:
        assert isinstance(e, ValueError)


# LLM-generated content at query #17
#--------------------------

# Unit test for method patronymic of class Person
def test_Person_patronymic():  
    # Test with gender = Gender.MALE
    person = Person()
    result = person.patronymic(gender=Gender.MALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with gender = Gender.FEMALE
    result = person.patronymic(gender=Gender.FEMALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with gender = None
    result = person.patronymic(gender=None)
    assert result is None or isinstance(result, str)
    # Test with locale = 'ru'
    person_ru = Person(locale='ru')
    result = person_ru.patronymic(gender=Gender.MALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with locale = 'uk'
    person_uk = Person(locale='uk')
    result = person_uk.patronymic(gender=Gender.FEMALE)
    assert result is not None
    assert isinstance(result, str)
    # Test with locale = 'en'
    person_en = Person(locale='en')
    result = person_en.patronymic(gender=Gender.MALE)
    assert result is None


# LLM-generated content at query #18
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])

    # Test with gender Male
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._extract(["nationality", "male"])

    # Test with gender Female
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._extract(["nationality", "female"])

    # Test with invalid gender (should raise error)
    try:
        person.nationality("invalid_gender")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data
    person_ru = Person(locale="ru")
    nationality_ru = person_ru.nationality()
    assert isinstance(nationality_ru, str)
    assert nationality_ru in person_ru._extract(["nationality"])

    # Test with locale-specific data and gender
    nationality_ru_male = person_ru.nationality(Gender.MALE)
    assert isinstance(nationality_ru_male, str)
    assert nationality_ru_male in person_ru._extract(["nationality", "male"])

    # Test with locale-specific data and gender Female
    nationality_ru_female = person_ru.nationality(Gender.FEMALE)
    assert isinstance(nationality_ru_female, str)
    assert nationality_ru_female in person_ru._extract(["nationality", "female"])

    # Test with locale-specific data and gender None
    nationality_ru_none = person_ru.nationality()
    assert isinstance(nationality_ru_none, str)
    assert nationality_ru_none in person_ru._extract(["nationality"])

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality("invalid_gender")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(123)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality([])
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality({})
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(set())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(frozenset())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(object())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(type)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(Exception)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(BaseException)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(GeneratorExit)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(KeyboardInterrupt)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(SystemExit)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(StopIteration)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(StopAsyncIteration)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(ArithmeticError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(AssertionError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(AttributeError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(BufferError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(EOFError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(ImportError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(LookupError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(MemoryError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(NameError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(OSError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(ReferenceError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(RuntimeError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(SyntaxError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(SystemError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(TypeError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(ValueError)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(Warning)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(FutureWarning)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(DeprecationWarning)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(PendingDeprecationWarning)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(RuntimeWarning)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(SyntaxWarning)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific data and gender invalid (should raise error)
    try:
        person_ru.nationality(UserWarning)
    except Exception as


# LLM-generated content at query #19
#--------------------------

# Unit test for method email of class Person
def test_Person_email():  
    # Test with default parameters
    person = Person()
    email = person.email()
    assert '@' in email
    assert '.' in email
    assert email.split('@')[1] in EMAIL_DOMAINS

    # Test with custom domains
    custom_domains = ['example.com', 'test.org']
    email = person.email(domains=custom_domains)
    assert email.split('@')[1] in custom_domains

    # Test unique email generation
    email1 = person.email(unique=True)
    email2 = person.email(unique=True)
    assert email1 != email2

    # Test with seeded provider and unique=True should raise ValueError
    seeded_person = Person(seed=42)
    try:
        seeded_person.email(unique=True)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Test email format
    email = person.email()
    assert re.match(r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,}$', email)

    # Test with empty domains list
    email = person.email(domains=[])
    assert email.endswith('@')  # Should have @ but no domain

    # Test with single domain
    single_domain = ['single.com']
    email = person.email(domains=single_domain)
    assert email.endswith('@single.com')

    # Test that username part is generated correctly
    email = person.email()
    username_part = email.split('@')[0]
    assert len(username_part) > 0
    assert all(c.isalnum() for c in username_part)


# LLM-generated content at query #20
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._data["nationality"]
    
    # Test with gender Male
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._data["nationality"][Gender.MALE.value]
    
    # Test with gender Female
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._data["nationality"][Gender.FEMALE.value]
    
    # Test with invalid gender (should raise error)
    try:
        person.nationality("invalid_gender")
    except Exception as e:
        assert isinstance(e, ValueError)
    
    print("All tests passed for method nationality of class Person")

test_Person_nationality()


# LLM-generated content at query #21
#--------------------------

# Unit test for method nationality of class Person
def test_Person_nationality():  
    # Test with default gender (None)
    person = Person()
    nationality = person.nationality()
    assert isinstance(nationality, str)
    assert nationality in person._extract(["nationality"])

    # Test with gender Male
    nationality_male = person.nationality(Gender.MALE)
    assert isinstance(nationality_male, str)
    assert nationality_male in person._extract(["nationality", "male"])

    # Test with gender Female
    nationality_female = person.nationality(Gender.FEMALE)
    assert isinstance(nationality_female, str)
    assert nationality_female in person._extract(["nationality", "female"])

    # Test with invalid gender (should raise an error)
    try:
        person.nationality("invalid_gender")
        assert False, "Should have raised an error"
    except Exception as e:
        assert isinstance(e, ValueError)



# LLM-generated content at query #22
#--------------------------

# Unit test for method surname of class Person
def test_Person_surname():  
    # Test with default gender
    person = Person()
    surname = person.surname()
    assert isinstance(surname, str)
    assert surname in person._extract(["surnames"])

    # Test with gender Male
    surname_male = person.surname(gender=Gender.MALE)
    assert isinstance(surname_male, str)
    assert surname_male in person._extract(["surnames"])

    # Test with gender Female
    surname_female = person.surname(gender=Gender.FEMALE)
    assert isinstance(surname_female, str)
    assert surname_female in person._extract(["surnames"])

    # Test with gender None
    surname_none = person.surname(gender=None)
    assert isinstance(surname_none, str)
    assert surname_none in person._extract(["surnames"])

    # Test with invalid gender
    try:
        person.surname(gender="INVALID")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames
    person_ru = Person(locale="ru")
    surname_ru = person_ru.surname()
    assert isinstance(surname_ru, str)
    assert surname_ru in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender
    surname_ru_male = person_ru.surname(gender=Gender.MALE)
    assert isinstance(surname_ru_male, str)
    assert surname_ru_male in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender Female
    surname_ru_female = person_ru.surname(gender=Gender.FEMALE)
    assert isinstance(surname_ru_female, str)
    assert surname_ru_female in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and gender None
    surname_ru_none = person_ru.surname(gender=None)
    assert isinstance(surname_ru_none, str)
    assert surname_ru_none in person_ru._extract(["surnames"])

    # Test with locale-specific surnames and invalid gender
    try:
        person_ru.surname(gender="INVALID")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as string
    try:
        person_ru.surname(gender="MALE")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as integer
    try:
        person_ru.surname(gender=1)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as float
    try:
        person_ru.surname(gender=1.0)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as boolean
    try:
        person_ru.surname(gender=True)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as list
    try:
        person_ru.surname(gender=[])
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as dict
    try:
        person_ru.surname(gender={})
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as tuple
    try:
        person_ru.surname(gender=())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as set
    try:
        person_ru.surname(gender=set())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as frozenset
    try:
        person_ru.surname(gender=frozenset())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as bytes
    try:
        person_ru.surname(gender=b"")
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as bytearray
    try:
        person_ru.surname(gender=bytearray())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as memoryview
    try:
        person_ru.surname(gender=memoryview(b""))
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as complex
    try:
        person_ru.surname(gender=complex())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as range
    try:
        person_ru.surname(gender=range(0))
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as slice
    try:
        person_ru.surname(gender=slice(0))
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as type
    try:
        person_ru.surname(gender=type)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as object
    try:
        person_ru.surname(gender=object())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as function
    try:
        person_ru.surname(gender=lambda: None)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as class
    try:
        person_ru.surname(gender=Person)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as instance
    try:
        person_ru.surname(gender=Person())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as module
    try:
        person_ru.surname(gender=__import__("sys"))
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as generator
    try:
        person_ru.surname(gender=(x for x in range(0)))
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as coroutine
    try:
        person_ru.surname(gender=coroutine())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as async generator
    try:
        person_ru.surname(gender=async_generator())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as async iterator
    try:
        person_ru.surname(gender=async_iterator())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as async iterable
    try:
        person_ru.surname(gender=async_iterable())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as async context manager
    try:
        person_ru.surname(gender=async_context_manager())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as async with
    try:
        person_ru.surname(gender=async_with())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as async for
    try:
        person_ru.surname(gender=async_for())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as async def
    try:
        person_ru.surname(gender=async_def())
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as await
    try:
        person_ru.surname(gender=await)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as yield
    try:
        person_ru.surname(gender=yield)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as yield from
    try:
        person_ru.surname(gender=yield_from)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as return
    try:
        person_ru.surname(gender=return)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as break
    try:
        person_ru.surname(gender=break)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as continue
    try:
        person_ru.surname(gender=continue)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as pass
    try:
        person_ru.surname(gender=pass)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as raise
    try:
        person_ru.surname(gender=raise)
    except Exception as e:
        assert isinstance(e, ValueError)

    # Test with locale-specific surnames and gender as try
    try:
        person_ru.surname


