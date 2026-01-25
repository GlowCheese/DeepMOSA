####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    number = payment.credit_card_number()
    assert len(number.replace(" ", "")) == 16
    assert number[:1] in {'4', '2', '5'}

def test_credit_card_number_visa():
    payment = Payment()
    number = payment.credit_card_number(CardType.VISA)
    assert len(number.replace(" ", "")) == 16
    assert number.startswith('4')

def test_credit_card_number_mastercard():
    payment = Payment()
    number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(number.replace(" ", "")) == 16
    assert number[:1] in {'2', '5'}

def test_credit_card_number_american_express():
    payment = Payment()
    number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(number.replace(" ", "")) == 15
    assert number.startswith('34') or number.startswith('37')

def test_credit_card_number_invalid_card_type():
    payment = Payment()
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_credit_card_number_visa():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert any(card_number.startswith(prefix) for prefix in ["2221", "2720", "5100", "5599"])

def test_credit_card_number_american_express():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert any(card_number.startswith(prefix) for prefix in ["34", "37"])

def test_credit_card_number_default():
    payment = Payment()
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_invalid_card_type():
    payment = Payment()
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        assert True


# LLM-generated content at query #3
#--------------------------

```python
def test_credit_card_number_visa():
    payment = Payment()
    number = payment.credit_card_number(CardType.VISA)
    assert len(number.replace(" ", "")) == 16
    assert number.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(number.replace(" ", "")) == 16
    assert number.startswith("22") or number.startswith("27") or number.startswith("51") or number.startswith("55")

def test_credit_card_number_american_express():
    payment = Payment()
    number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(number.replace(" ", "")) == 15
    assert number.startswith("34") or number.startswith("37")

def test_credit_card_number_default():
    payment = Payment()
    number = payment.credit_card_number()
    assert len(number.replace(" ", "")) == 16

def test_credit_card_number_luhn_check():
    payment = Payment()
    number = payment.credit_card_number()
    digits = number.replace(" ", "")
    check_digit = int(digits[-1])
    calculated_check_digit = int(luhn_checksum(digits[:-1]))
    assert check_digit == calculated_check_digit

def test_credit_card_number_unsupported_card_type():
    payment = Payment()
    try:
        payment.credit_card_number("UnsupportedCardType")
        assert False, "Expected NotImplementedError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #4
#--------------------------

def test_credit_card_number_visa():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert (2221 <= int(card_number[:4]) <= 2720) or (5100 <= int(card_number[:4]) <= 5599)

def test_credit_card_number_amex():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

def test_credit_card_number_default():
    payment = Payment()
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) in (15, 16)

def test_credit_card_number_luhn_valid():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    digits = card_number.replace(" ", "")
    check_digit = int(digits[-1])
    checksum = luhn_checksum(digits[:-1])
    assert check_digit == int(checksum)


# LLM-generated content at query #5
#--------------------------

```python
def test_credit_card_number_visa():
    payment = Payment()
    number = payment.credit_card_number(CardType.VISA)
    assert len(number.replace(" ", "")) == 16
    assert number.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(number.replace(" ", "")) == 16
    assert number[:2] in ["22", "23", "24", "25", "26", "27", "51", "52", "53", "54", "55"]

def test_credit_card_number_american_express():
    payment = Payment()
    number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(number.replace(" ", "")) == 15
    assert number.startswith("34") or number.startswith("37")

def test_credit_card_number_default():
    payment = Payment()
    number = payment.credit_card_number()
    assert len(number.replace(" ", "")) == 16
    assert number.startswith("4") or number[:2] in ["22", "23", "24", "25", "26", "27", "51", "52", "53", "54", "55"]

def test_credit_card_number_valid_luhn():
    payment = Payment()
    number = payment.credit_card_number()
    digits = [int(d) for d in number.replace(" ", "")]
    check_digit = digits.pop()
    digits.reverse()
    doubled = [digits[i] * 2 if i % 2 == 0 else digits[i] for i in range(len(digits))]
    summed = [d - 9 if d > 9 else d for d in doubled]
    total = sum(summed) + check_digit
    assert total % 10 == 0


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_credit_card_number_visa():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert (2221 <= int(card_number[:4]) <= 2720) or (5100 <= int(card_number[:4]) <= 5599)

def test_credit_card_number_amex():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

def test_credit_card_number_default():
    payment = Payment()
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) in (15, 16)

def test_credit_card_number_luhn_valid():
    payment = Payment()
    card_number = payment.credit_card_number()
    digits = card_number.replace(" ", "")
    check_digit = int(digits[-1])
    checksum = luhn_checksum(digits[:-1])
    assert check_digit == int(checksum)

def test_credit_card_number_unsupported_type():
    payment = Payment()
    try:
        payment.credit_card_number("unsupported_type")
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #2
#--------------------------

def test_credit_card_number_visa():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert (2221 <= int(card_number[:4]) <= 2720) or (5100 <= int(card_number[:4]) <= 5599)

def test_credit_card_number_amex():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

def test_credit_card_number_default():
    payment = Payment()
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) in (15, 16)

def test_credit_card_number_luhn_valid():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    digits = card_number.replace(" ", "")
    check = 0
    for i, s in enumerate(reversed(digits)):
        sx = int(s)
        if i % 2 == 1:
            sx *= 2
        if sx > 9:
            sx -= 9
        check += sx
    assert check % 10 == 0


# LLM-generated content at query #3
#--------------------------

```python
def test_credit_card_number_no_additional_digits_needed():
    payment = Payment(seed=12345)
    card_type = CardType.AMERICAN_EXPRESS
    card_number = payment.credit_card_number(card_type)
    assert len(card_number.replace(" ", "")) == 15


# LLM-generated content at query #4
#--------------------------

```python
def test_credit_card_number_length():
    payment = Payment()
    card_types = [CardType.VISA, CardType.MASTER_CARD, CardType.AMERICAN_EXPRESS]
    for card_type in card_types:
        card_number = payment.credit_card_number(card_type)
        card_digits = card_number.replace(" ", "")
        if card_type == CardType.AMERICAN_EXPRESS:
            assert len(card_digits) == 15
        else:
            assert len(card_digits) == 16


# LLM-generated content at query #5
#--------------------------

```python
def test_credit_card_number_visa():
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_master_card():
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("22") or card_number.startswith("27") or card_number.startswith("51") or card_number.startswith("55")

def test_credit_card_number_american_express():
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

def test_credit_card_number_default():
    payment = Payment()
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4") or card_number.startswith("22") or card_number.startswith("27") or card_number.startswith("51") or card_number.startswith("55")


# LLM-generated content at query #6
#--------------------------

```python
def test_credit_card_number_visa_does_not_enter_while_loop():
    payment = Payment(seed=42)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16


# LLM-generated content at query #7
#--------------------------

def test_credit_card_number_with_visa_does_not_enter_while_loop():
    payment = Payment(seed=42)
    visa_card = payment.credit_card_number(card_type=CardType.VISA)
    assert len(visa_card.replace(" ", "")) == 16

def test_credit_card_number_with_mastercard_does_not_enter_while_loop():
    payment = Payment(seed=42)
    mastercard = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert len(mastercard.replace(" ", "")) == 16

def test_credit_card_number_with_amex_does_not_enter_while_loop():
    payment = Payment(seed=42)
    amex_card = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15


# LLM-generated content at query #8
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number[:1] in ["4"]

def test_credit_card_number_visa():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number[:1] in ["4"]

def test_credit_card_number_mastercard():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number[:2] in ["22", "23", "24", "25", "26", "27", "51", "52", "53", "54", "55"]

def test_credit_card_number_american_express():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number[:2] in ["34", "37"]

def test_credit_card_number_unsupported_card_type():
    payment = Payment()
    try:
        payment.credit_card_number("UnsupportedCardType")
    except NonEnumerableError:
        pass
    else:
        assert False, "Expected NonEnumerableError"


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_credit_card_number_visa():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert (2221 <= int(card_number[:4]) <= 2720) or (5100 <= int(card_number[:4]) <= 5599)

def test_credit_card_number_amex():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

def test_credit_card_number_default():
    payment = Payment()
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) in (15, 16)

def test_credit_card_number_luhn_valid():
    payment = Payment()
    for _ in range(10):  # Test multiple cards
        card_number = payment.credit_card_number()
        digits = card_number.replace(" ", "")
        check_digit = int(digits[-1])
        checksum = 0
        
        for i, d in enumerate(reversed(digits[:-1])):
            d = int(d)
            if i % 2 == 0:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
        
        assert (checksum + check_digit) % 10 == 0


# LLM-generated content at query #2
#--------------------------

def test_credit_card_number_with_visa_does_not_enter_while_loop():
    payment = Payment()
    payment.random = Mock()
    payment.random.choice_enum_item.return_value = CardType.VISA
    payment.random.randint.return_value = 4000  # Will produce "4000" which is length 4
    payment.random.choice.return_value = '0'  # Doesn't matter since loop shouldn't run
    result = payment.credit_card_number()
    assert len(result.split()[0]) == 4  # First group should be exactly 4 digits (4000)


# LLM-generated content at query #3
#--------------------------

```python
def test_credit_card_number_length():
    payment = Payment()
    card_types = [CardType.VISA, CardType.MASTER_CARD, CardType.AMERICAN_EXPRESS]
    for card_type in card_types:
        card_number = payment.credit_card_number(card_type=card_type)
        if card_type == CardType.AMERICAN_EXPRESS:
            assert len(card_number.replace(" ", "")) == 15
        else:
            assert len(card_number.replace(" ", "")) == 16


# LLM-generated content at query #4
#--------------------------

```python
def test_credit_card_number_predicate_evaluates_to_false():
    payment = Payment()
    card_type = CardType.VISA
    number = payment.random.randint(4000, 4999)
    str_num = str(number)
    assert len(str_num) >= 15


# LLM-generated content at query #5
#--------------------------

def test_credit_card_number_visa_does_not_enter_while_loop():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_mastercard_does_not_enter_while_loop():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_amex_does_not_enter_while_loop():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15


# LLM-generated content at query #6
#--------------------------

```python
def test_credit_card_number_length_condition():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    str_num = card_number.replace(" ", "")
    assert len(str_num) == 16


# LLM-generated content at query #7
#--------------------------

```python
def test_credit_card_number_visa():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("22") or card_number.startswith("27") or card_number.startswith("51") or card_number.startswith("55")

def test_credit_card_number_american_express():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

def test_credit_card_number_default():
    payment = Payment()
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4") or card_number.startswith("22") or card_number.startswith("27") or card_number.startswith("51") or card_number.startswith("55")

def test_credit_card_number_invalid_card_type():
    payment = Payment()
    try:
        payment.credit_card_number("InvalidCardType")
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #8
#--------------------------

```python
def test_credit_card_number_with_valid_card_type():
    payment = Payment()
    card_types = [CardType.VISA, CardType.MASTER_CARD, CardType.AMERICAN_EXPRESS]
    for card_type in card_types:
        card_number = payment.credit_card_number(card_type)
        assert isinstance(card_number, str)
        assert len(card_number.replace(" ", "")) in [15, 16]


