####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")
    assert result.count(" ") == 3

def test_credit_card_number_master_card():
    payment = Payment()
    result = payment.credit_card_number(CardType.MASTER_CARD)
    number = result.replace(" ", "")
    assert len(number) == 16
    first_four = int(number[:4])
    assert (2221 <= first_four <= 2720) or (5100 <= first_four <= 5599)
    assert result.count(" ") == 3

def test_credit_card_number_american_express():
    payment = Payment()
    result = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    number = result.replace(" ", "")
    assert len(number) == 15
    assert number.startswith("34") or number.startswith("37")
    assert result.count(" ") == 2

def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert len(result.replace(" ", "")) in [15, 16]
    assert result.count(" ") in [2, 3]

def test_credit_card_number_luhn_valid():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    number = result.replace(" ", "")
    check = 0
    for i, s in enumerate(reversed(number)):
        sx = int(s)
        if i % 2 == 0:
            sx *= 2
        if sx > 9:
            sx -= 9
        check += sx
    assert check % 10 == 0

def test_credit_card_number_invalid_card_type():
    payment = Payment()
    try:
        payment.credit_card_number("InvalidType")
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #2
#--------------------------

def test_credit_card_number_visa_length_condition_false():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment(seed=42)
    result = payment.credit_card_number(card_type=CardType.VISA)
    assert len(result.replace(" ", "")) == 16


# LLM-generated content at query #3
#--------------------------

def test_credit_card_number_visa_length_condition_false():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.VISA)
    number_part = result.replace(" ", "")
    assert len(number_part) == 16
    assert number_part.startswith("4")


# LLM-generated content at query #4
#--------------------------

def test_credit_card_number_visa():
    payment = Payment()
    number = payment.credit_card_number(CardType.VISA)
    parts = number.split()
    assert len(parts) == 4
    assert all(len(part) == 4 for part in parts)
    assert parts[0].startswith('4')
    digits = ''.join(parts)
    assert len(digits) == 16
    check_digit = int(digits[-1])
    calculated_check = int(luhn_checksum(digits[:-1]))
    assert check_digit == calculated_check

def test_credit_card_number_master_card():
    payment = Payment()
    number = payment.credit_card_number(CardType.MASTER_CARD)
    parts = number.split()
    assert len(parts) == 4
    assert all(len(part) == 4 for part in parts)
    digits = ''.join(parts)
    assert len(digits) == 16
    first_four = int(digits[:4])
    assert (2221 <= first_four <= 2720) or (5100 <= first_four <= 5599)
    check_digit = int(digits[-1])
    calculated_check = int(luhn_checksum(digits[:-1]))
    assert check_digit == calculated_check

def test_credit_card_number_american_express():
    payment = Payment()
    number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    parts = number.split()
    assert len(parts) == 3
    assert len(parts[0]) == 4
    assert len(parts[1]) == 6
    assert len(parts[2]) == 5
    digits = ''.join(parts)
    assert len(digits) == 15
    first_two = int(digits[:2])
    assert first_two in [34, 37]
    check_digit = int(digits[-1])
    calculated_check = int(luhn_checksum(digits[:-1]))
    assert check_digit == calculated_check

def test_credit_card_number_default_random():
    payment = Payment()
    number = payment.credit_card_number()
    parts = number.split()
    digits = ''.join(parts)
    assert len(digits) in [15, 16]
    if len(digits) == 16:
        assert parts[0].startswith('4') or (2221 <= int(parts[0]) <= 2720) or (5100 <= int(parts[0]) <= 5599)
    else:
        assert len(parts) == 3
        assert int(digits[:2]) in [34, 37]
    check_digit = int(digits[-1])
    calculated_check = int(luhn_checksum(digits[:-1]))
    assert check_digit == calculated_check

def test_credit_card_number_unsupported_type_raises_error():
    payment = Payment()
    try:
        payment.credit_card_number("unsupported")
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #5
#--------------------------

def test_credit_card_number_visa_length_condition_false():
    payment = Payment(seed=12345)
    card_type = CardType.VISA
    result = payment.credit_card_number(card_type=card_type)
    number_part = result.replace(" ", "")
    assert len(number_part) == 16
    assert number_part.startswith("4")


# LLM-generated content at query #6
#--------------------------

def test_credit_card_number_visa():
    payment = Payment()
    number = payment.credit_card_number(CardType.VISA)
    parts = number.replace(" ", "")
    assert parts.startswith("4")
    assert len(parts) == 16
    assert luhn_checksum(parts[:-1]) == parts[-1]

def test_credit_card_number_mastercard():
    payment = Payment()
    number = payment.credit_card_number(CardType.MASTER_CARD)
    parts = number.replace(" ", "")
    assert len(parts) == 16
    first_four = int(parts[:4])
    assert (2221 <= first_four <= 2720) or (5100 <= first_four <= 5599)
    assert luhn_checksum(parts[:-1]) == parts[-1]

def test_credit_card_number_american_express():
    payment = Payment()
    number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    parts = number.replace(" ", "")
    assert len(parts) == 15
    first_two = int(parts[:2])
    assert first_two in [34, 37]
    assert luhn_checksum(parts[:-1]) == parts[-1]

def test_credit_card_number_default_random():
    payment = Payment()
    number = payment.credit_card_number()
    parts = number.replace(" ", "")
    assert len(parts) in [15, 16]
    assert luhn_checksum(parts[:-1]) == parts[-1]

def test_credit_card_number_unsupported_type():
    payment = Payment()
    try:
        payment.credit_card_number("unsupported")
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #7
#--------------------------

def test_credit_card_number_visa_length_condition_false():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    parts = card_number.replace(" ", "")
    assert len(parts) == 16
    assert parts[0] == "4"

def test_credit_card_number_mastercard_length_condition_false():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    parts = card_number.replace(" ", "")
    assert len(parts) == 16
    first_four = int(parts[:4])
    assert (2221 <= first_four <= 2720) or (5100 <= first_four <= 5599)

def test_credit_card_number_american_express_length_condition_false():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    parts = card_number.replace(" ", "")
    assert len(parts) == 15
    first_two = int(parts[:2])
    assert first_two in [34, 37]


# LLM-generated content at query #8
#--------------------------

def test_credit_card_number_visa_length_condition_false():
    from mimesis import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    parts = card_number.replace(" ", "")
    assert len(parts) == 16
    assert parts[0] == "4"

def test_credit_card_number_mastercard_length_condition_false():
    from mimesis import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    parts = card_number.replace(" ", "")
    assert len(parts) == 16
    first_four = int(parts[:4])
    assert (2221 <= first_four <= 2720) or (5100 <= first_four <= 5599)

def test_credit_card_number_american_express_length_condition_false():
    from mimesis import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    parts = card_number.replace(" ", "")
    assert len(parts) == 15
    first_two = int(parts[:2])
    assert first_two in [34, 37]


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    assert result is not None
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")

def test_credit_card_number_master_card():
    payment = Payment()
    result = payment.credit_card_number(CardType.MASTER_CARD)
    assert result is not None
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    first_two = int(result[:2])
    first_four = int(result[:4])
    assert (first_two >= 51 and first_two <= 55) or (first_four >= 2221 and first_four <= 2720)

def test_credit_card_number_american_express():
    payment = Payment()
    result = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert result is not None
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 15
    assert result.startswith("34") or result.startswith("37")

def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert result is not None
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) in [15, 16]

def test_credit_card_number_invalid_card_type():
    payment = Payment()
    try:
        payment.credit_card_number("InvalidType")
        assert False
    except NonEnumerableError:
        assert True

def test_credit_card_number_luhn_checksum():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    digits = result.replace(" ", "")
    check_digit = int(digits[-1])
    total = 0
    for i, digit in enumerate(reversed(digits[:-1])):
        n = int(digit)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    expected_check_digit = (total * 9) % 10
    assert check_digit == expected_check_digit


# LLM-generated content at query #2
#--------------------------

def test_credit_card_number_visa_length_condition_false():
    payment = Payment(seed=12345)
    result = payment.credit_card_number(card_type=CardType.VISA)
    assert len(result.replace(" ", "")) == 16


# LLM-generated content at query #3
#--------------------------

def test_credit_card_number_visa_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_mastercard_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_american_express_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15


# LLM-generated content at query #4
#--------------------------

def test_credit_card_number_visa_no_while_loop():
    payment = Payment(seed=42)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_mastercard_no_while_loop():
    payment = Payment(seed=42)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_american_express_no_while_loop():
    payment = Payment(seed=42)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15


# LLM-generated content at query #5
#--------------------------

def test_credit_card_number_visa_length_condition_false():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    number_part = card_number.replace(" ", "")
    assert len(number_part) == 16


# LLM-generated content at query #6
#--------------------------

def test_credit_card_number_visa_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_mastercard_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_american_express_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15


# LLM-generated content at query #7
#--------------------------

def test_credit_card_number_visa_length_condition_false():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment(seed=12345)
    result = payment.credit_card_number(card_type=CardType.VISA)
    number_part = result.replace(" ", "")
    assert len(number_part) == 16
    assert number_part.startswith("4")


# LLM-generated content at query #8
#--------------------------

def test_credit_card_number_visa_length_condition_false():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    groups = card_number.split()
    first_group = groups[0]
    number = int(first_group)
    assert 4000 <= number <= 4999
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_mastercard_length_condition_false():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    groups = card_number.split()
    first_group = groups[0]
    number = int(first_group)
    assert (2221 <= number <= 2720) or (5100 <= number <= 5599)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_american_express_length_condition_false():
    from mimesis.providers.payment import Payment
    from mimesis.enums import CardType
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    groups = card_number.split()
    first_group = groups[0]
    number = int(first_group)
    assert number in [34, 37]
    assert len(card_number.replace(" ", "")) == 15


