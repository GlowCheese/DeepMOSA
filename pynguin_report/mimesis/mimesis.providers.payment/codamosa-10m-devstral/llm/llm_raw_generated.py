####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment(locale=Locale.EN)

    # Test default (random card type)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in (15, 16)
    assert all(c.isdigit() or c.isspace() for c in card_number)

    # Test Visa
    card_number = payment.credit_card_number(CardType.VISA)
    assert card_number.startswith("4")
    assert len(card_number.replace(" ", "")) == 16

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert card_number.startswith("2") or card_number.startswith("5")
    assert len(card_number.replace(" ", "")) == 16

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert card_number.startswith("34") or card_number.startswith("37")
    assert len(card_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("INVALID_TYPE")


# LLM-generated content at query #2
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default case (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))


# LLM-generated content at query #3
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #4
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))


# LLM-generated content at query #5
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #6
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #7
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment(locale=Locale.EN)

    # Test default card type (random)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in [15, 16]
    assert all(c.isdigit() or c.isspace() for c in card_number)

    # Test Visa card type
    visa_card = payment.credit_card_number(card_type=CardType.VISA)
    assert visa_card.startswith("4")
    assert len(visa_card.replace(" ", "")) == 16

    # Test MasterCard card type
    mastercard = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert mastercard.startswith(("2", "5"))
    assert len(mastercard.replace(" ", "")) == 16

    # Test American Express card type
    amex_card = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert amex_card.startswith(("34", "37"))
    assert len(amex_card.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(card_type="InvalidCardType")


# LLM-generated content at query #8
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default case (Visa)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number) == 19  # 4 groups of 4 digits with spaces
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert isinstance(card_number, str)
    assert len(card_number) == 19
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert isinstance(card_number, str)
    assert len(card_number) == 19  # 3 groups (4, 6, 5 digits) with spaces
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))


# LLM-generated content at query #9
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (random)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in [15, 16]
    assert all(c.isdigit() or c.isspace() for c in card_number)

    # Test Visa
    card_number = payment.credit_card_number(CardType.VISA)
    assert card_number.startswith("4")
    assert len(card_number.replace(" ", "")) == 16

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert card_number.startswith(("2", "5"))
    assert len(card_number.replace(" ", "")) == 16

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert card_number.startswith(("34", "37"))
    assert len(card_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("INVALID_CARD_TYPE")


# LLM-generated content at query #10
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment(locale=Locale.EN)

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #11
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType.DISCOVER)

    # Test Luhn checksum
    card_number = payment.credit_card_number().replace(" ", "")
    total = 0
    for i, digit in enumerate(reversed(card_number)):
        digit = int(digit)
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit = digit - 9
        total += digit
    assert total % 10 == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment(locale=Locale.EN)

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^4\d{3} \d{4} \d{4} \d{4}$", card_number)
    assert len(card_number.replace(" ", "")) == 16

    # Test Visa
    card_number = payment.credit_card_number(CardType.VISA)
    assert re.match(r"^4\d{3} \d{4} \d{4} \d{4}$", card_number)
    assert len(card_number.replace(" ", "")) == 16

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^(2[2-7]\d{2}|5[1-5]\d{2}) \d{4} \d{4} \d{4}$", card_number)
    assert len(card_number.replace(" ", "")) == 16

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^(34|37) \d{6} \d{5}$", card_number)
    assert len(card_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #13
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default (random card type)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in [15, 16]
    assert all(c.isdigit() or c.isspace() for c in card_number)

    # Test Visa
    visa_number = payment.credit_card_number(CardType.VISA)
    assert visa_number.startswith("4")
    assert len(visa_number.replace(" ", "")) == 16

    # Test MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert mastercard_number.startswith(("2", "5"))
    assert len(mastercard_number.replace(" ", "")) == 16

    # Test American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert amex_number.startswith(("34", "37"))
    assert len(amex_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("invalid_type")


# LLM-generated content at query #14
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test Visa
    card_number = payment.credit_card_number(CardType.VISA)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #15
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))


# LLM-generated content at query #16
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (random)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in {15, 16}
    assert card_number.count(" ") in {3, 4}

    # Test Visa
    card_number = payment.credit_card_number(CardType.VISA)
    assert card_number.startswith("4")
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.count(" ") == 3

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert card_number.startswith(("2", "5"))
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.count(" ") == 3

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert card_number.startswith(("34", "37"))
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.count(" ") == 2

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #17
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default case (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #18
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default behavior (random card type)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in [15, 16]
    assert all(c.isdigit() or c.isspace() for c in card_number)

    # Test Visa
    visa_number = payment.credit_card_number(CardType.VISA)
    assert visa_number.startswith("4")
    assert len(visa_number.replace(" ", "")) == 16

    # Test MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert mastercard_number.startswith(("2", "5"))
    assert len(mastercard_number.replace(" ", "")) == 16

    # Test American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert amex_number.startswith(("34", "37"))
    assert len(amex_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #19
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #20
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default case (no card_type specified)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)

    # Test Visa card
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert card_number.startswith("4")
    assert len(card_number.replace(" ", "")) == 16

    # Test MasterCard
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert card_number.startswith("2") or card_number.startswith("5")
    assert len(card_number.replace(" ", "")) == 16

    # Test American Express
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert card_number.startswith("34") or card_number.startswith("37")
    assert len(card_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(card_type="INVALID_TYPE")


# LLM-generated content at query #21
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))


# LLM-generated content at query #22
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))


# LLM-generated content at query #23
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))

    # Test Luhn checksum
    card_number = payment.credit_card_number().replace(" ", "")
    total = 0
    for i, digit in enumerate(reversed(card_number)):
        digit = int(digit)
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    assert total % 10 == 0


# LLM-generated content at query #24
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #25
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #26
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default case (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(card_type="InvalidCardType")


# LLM-generated content at query #27
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))


# LLM-generated content at query #28
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(card_type="InvalidCardType")


# LLM-generated content at query #29
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default (random card type)
    card = payment.credit_card_number()
    assert isinstance(card, str)
    assert len(card.replace(" ", "")) in [15, 16]
    assert card.count(" ") in [2, 3]

    # Test Visa
    card = payment.credit_card_number(CardType.VISA)
    assert card.startswith("4")
    assert len(card.replace(" ", "")) == 16
    assert card.count(" ") == 3

    # Test MasterCard
    card = payment.credit_card_number(CardType.MASTER_CARD)
    assert card.startswith(("2", "5"))
    assert len(card.replace(" ", "")) == 16
    assert card.count(" ") == 3

    # Test American Express
    card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert card.startswith(("34", "37"))
    assert len(card.replace(" ", "")) == 15
    assert card.count(" ") == 2

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("INVALID_TYPE")


# LLM-generated content at query #30
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #2
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default behavior (random card type)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in [15, 16]
    assert all(c.isdigit() or c.isspace() for c in card_number)

    # Test Visa card
    visa_card = payment.credit_card_number(CardType.VISA)
    assert visa_card.startswith("4")
    assert len(visa_card.replace(" ", "")) == 16

    # Test MasterCard
    mastercard = payment.credit_card_number(CardType.MASTER_CARD)
    assert mastercard.startswith(("2", "5"))
    assert len(mastercard.replace(" ", "")) == 16

    # Test American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert amex_card.startswith(("34", "37"))
    assert len(amex_card.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #3
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #4
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #5
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default case (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))


# LLM-generated content at query #6
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default case (random card type)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in {15, 16}  # AMEX is 15, others 16
    assert all(c.isdigit() or c.isspace() for c in card_number)

    # Test Visa
    card_number = payment.credit_card_number(CardType.VISA)
    assert card_number.startswith("4")
    assert len(card_number.replace(" ", "")) == 16

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert card_number.startswith(("2", "5"))
    assert len(card_number.replace(" ", "")) == 16

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert card_number.startswith(("34", "37"))
    assert len(card_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #7
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment(locale=Locale.EN)
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #8
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #9
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment(locale=Locale.EN)

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(card_type="InvalidCardType")

    # Test Luhn checksum
    def luhn_check(card_number):
        card_number = card_number.replace(" ", "")
        total = 0
        for i, digit in enumerate(reversed(card_number)):
            digit = int(digit)
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit = digit - 9
            total += digit
        return total % 10 == 0

    for _ in range(10):
        card_number = payment.credit_card_number()
        assert luhn_check(card_number)


# LLM-generated content at query #10
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith(("2", "5"))

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith(("34", "37"))

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType.DISCOVER)


# LLM-generated content at query #11
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(card_type="InvalidCardType")

    # Test Luhn checksum
    card_number_clean = card_number.replace(" ", "")
    total = 0
    for i, digit in enumerate(reversed(card_number_clean)):
        digit = int(digit)
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit = digit - 9
        total += digit
    assert total % 10 == 0


# LLM-generated content at query #12
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default case (no card_type specified)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in [15, 16]
    assert all(c.isdigit() or c.isspace() for c in card_number)

    # Test Visa card
    card_number = payment.credit_card_number(CardType.VISA)
    assert card_number.startswith("4")
    assert len(card_number.replace(" ", "")) == 16

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert card_number.startswith(("2", "5"))
    assert len(card_number.replace(" ", "")) == 16

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert card_number.startswith(("34", "37"))
    assert len(card_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #13
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default (random card type)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in [15, 16]
    assert all(c.isdigit() or c.isspace() for c in card_number)

    # Test Visa
    visa_number = payment.credit_card_number(CardType.VISA)
    assert visa_number.startswith("4")
    assert len(visa_number.replace(" ", "")) == 16

    # Test MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert mastercard_number.startswith(("2", "5"))
    assert len(mastercard_number.replace(" ", "")) == 16

    # Test American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert amex_number.startswith(("34", "37"))
    assert len(amex_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #14
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment(locale=Locale.EN)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in {15, 16}
    assert all(c.isdigit() or c.isspace() for c in card_number)

    card_number_visa = payment.credit_card_number(card_type=CardType.VISA)
    assert card_number_visa.startswith("4")
    assert len(card_number_visa.replace(" ", "")) == 16

    card_number_mastercard = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert card_number_mastercard.startswith(("2", "5"))
    assert len(card_number_mastercard.replace(" ", "")) == 16

    card_number_amex = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert card_number_amex.startswith(("34", "37"))
    assert len(card_number_amex.replace(" ", "")) == 15

    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(card_type=CardType("UNKNOWN"))


# LLM-generated content at query #15
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default case (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))

    # Test Luhn checksum
    card_number = payment.credit_card_number()
    digits = card_number.replace(" ", "")
    total = 0
    for i, digit in enumerate(reversed(digits)):
        num = int(digit)
        if i % 2 == 1:
            num *= 2
            if num > 9:
                num = num - 9
        total += num
    assert total % 10 == 0


# LLM-generated content at query #16
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default (random card type)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in [15, 16]
    assert card_number.count(" ") in [3, 4]

    # Test Visa
    visa_number = payment.credit_card_number(CardType.VISA)
    assert visa_number.startswith("4")
    assert len(visa_number.replace(" ", "")) == 16

    # Test MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert mastercard_number.startswith(("2", "5"))
    assert len(mastercard_number.replace(" ", "")) == 16

    # Test American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert amex_number.startswith(("34", "37"))
    assert len(amex_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))


# LLM-generated content at query #17
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))

    # Test Luhn checksum
    for _ in range(10):
        card_number = payment.credit_card_number()
        digits = card_number.replace(" ", "")
        total = 0
        for i, digit in enumerate(reversed(digits)):
            num = int(digit)
            if i % 2 == 1:
                num *= 2
                if num > 9:
                    num = num - 9
            total += num
        assert total % 10 == 0


# LLM-generated content at query #18
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")

    # Test Luhn checksum
    card_number = payment.credit_card_number()
    digits = card_number.replace(" ", "")
    total = 0
    for i, digit in enumerate(reversed(digits)):
        num = int(digit)
        if i % 2 == 1:
            num *= 2
            if num > 9:
                num = num - 9
        total += num
    assert total % 10 == 0


# LLM-generated content at query #19
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #20
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment(locale=Locale.EN)

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))

    # Test Luhn checksum
    card_number = payment.credit_card_number().replace(" ", "")
    total = 0
    for i, digit in enumerate(reversed(card_number)):
        digit = int(digit)
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit = digit - 9
        total += digit
    assert total % 10 == 0


# LLM-generated content at query #21
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #22
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #23
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #24
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #25
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default (random card type)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16  # Default length for Visa/MasterCard

    # Test Visa
    card_number = payment.credit_card_number(CardType.VISA)
    assert card_number.startswith("4")
    assert len(card_number.replace(" ", "")) == 16

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert card_number.startswith(("2", "5"))
    assert len(card_number.replace(" ", "")) == 16

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert card_number.startswith(("34", "37"))
    assert len(card_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #26
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #27
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default (random card type)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in [15, 16]
    assert all(c.isdigit() or c.isspace() for c in card_number)

    # Test Visa
    card_number = payment.credit_card_number(CardType.VISA)
    assert card_number.startswith("4")
    assert len(card_number.replace(" ", "")) == 16

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert card_number.startswith("2") or card_number.startswith("5")
    assert len(card_number.replace(" ", "")) == 16

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert card_number.startswith("3")
    assert len(card_number.replace(" ", "")) == 15

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #28
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("3")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number(CardType("INVALID"))


# LLM-generated content at query #29
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", card_number)
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", card_number)
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")


# LLM-generated content at query #30
#--------------------------

```python
def test_Payment_credit_card_number():
    payment = Payment()

    # Test default case (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

    # Test invalid card type
    with pytest.raises(NonEnumerableError):
        payment.credit_card_number("InvalidCardType")

    # Test Luhn checksum
    card_number = payment.credit_card_number()
    digits = card_number.replace(" ", "")
    total = 0
    for i, digit in enumerate(reversed(digits)):
        num = int(digit)
        if i % 2 == 1:
            num *= 2
            if num > 9:
                num = (num // 10) + (num % 10)
        total += num
    assert total % 10 == 0


