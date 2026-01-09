####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
    total = 0
    for i, digit in enumerate(reversed(digits[:-1])):
        n = int(digit)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    assert (total * 9) % 10 == check_digit

def test_credit_card_number_mastercard():
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
    total = 0
    for i, digit in enumerate(reversed(digits[:-1])):
        n = int(digit)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    assert (total * 9) % 10 == check_digit

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
    total = 0
    for i, digit in enumerate(reversed(digits[:-1])):
        n = int(digit)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    assert (total * 9) % 10 == check_digit

def test_credit_card_number_default_random():
    payment = Payment()
    number = payment.credit_card_number()
    parts = number.split()
    digits = ''.join(parts)
    assert len(digits) in [15, 16]
    if len(digits) == 16:
        first_digit = digits[0]
        assert first_digit in ['2', '3', '4', '5']
    else:
        first_two = digits[:2]
        assert first_two in ['34', '37']
    check_digit = int(digits[-1])
    total = 0
    for i, digit in enumerate(reversed(digits[:-1])):
        n = int(digit)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    assert (total * 9) % 10 == check_digit

def test_credit_card_number_unsupported_card_type():
    payment = Payment()
    try:
        payment.credit_card_number('unsupported')
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #2
#--------------------------

def test_credit_card_number_visa():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    groups = card_number.split()
    assert len(groups) == 4
    assert all(len(g) == 4 for g in groups)
    assert groups[0].startswith('4')
    number_without_spaces = ''.join(groups)
    assert len(number_without_spaces) == 16
    checksum = luhn_checksum(number_without_spaces[:-1])
    assert number_without_spaces[-1] == checksum

def test_credit_card_number_master_card():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    groups = card_number.split()
    assert len(groups) == 4
    assert all(len(g) == 4 for g in groups)
    first_group = int(groups[0])
    assert (2221 <= first_group <= 2720) or (5100 <= first_group <= 5599)
    number_without_spaces = ''.join(groups)
    assert len(number_without_spaces) == 16
    checksum = luhn_checksum(number_without_spaces[:-1])
    assert number_without_spaces[-1] == checksum

def test_credit_card_number_american_express():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    groups = card_number.split()
    assert len(groups) == 3
    assert len(groups[0]) == 4
    assert len(groups[1]) == 6
    assert len(groups[2]) == 5
    first_group = int(groups[0])
    assert first_group in [34, 37]
    number_without_spaces = ''.join(groups)
    assert len(number_without_spaces) == 15
    checksum = luhn_checksum(number_without_spaces[:-1])
    assert number_without_spaces[-1] == checksum

def test_credit_card_number_default_random():
    payment = Payment()
    card_number = payment.credit_card_number()
    groups = card_number.split()
    assert len(groups) in [3, 4]
    number_without_spaces = ''.join(groups)
    assert len(number_without_spaces) in [15, 16]
    checksum = luhn_checksum(number_without_spaces[:-1])
    assert number_without_spaces[-1] == checksum

def test_credit_card_number_invalid_card_type():
    payment = Payment()
    try:
        payment.credit_card_number("InvalidType")
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #3
#--------------------------

def test_credit_card_number_visa_no_while_loop():
    payment = Payment(seed=42)
    result = payment.credit_card_number(card_type=CardType.VISA)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")

def test_credit_card_number_master_card_no_while_loop():
    payment = Payment(seed=42)
    result = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    card_digits = result.replace(" ", "")
    assert len(card_digits) == 16
    assert int(card_digits[:4]) >= 2221 and int(card_digits[:4]) <= 2720 or int(card_digits[:4]) >= 5100 and int(card_digits[:4]) <= 5599

def test_credit_card_number_american_express_no_while_loop():
    payment = Payment(seed=42)
    result = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    card_digits = result.replace(" ", "")
    assert len(card_digits) == 15
    assert card_digits.startswith("34") or card_digits.startswith("37")


# LLM-generated content at query #4
#--------------------------

def test_credit_card_number_visa_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    parts = card_number.replace(" ", "")
    assert len(parts) == 16

def test_credit_card_number_mastercard_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    parts = card_number.replace(" ", "")
    assert len(parts) == 16

def test_credit_card_number_american_express_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    parts = card_number.replace(" ", "")
    assert len(parts) == 15


# LLM-generated content at query #5
#--------------------------

def test_credit_card_number_visa_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    parts = card_number.replace(" ", "")
    assert len(parts) == 16

def test_credit_card_number_mastercard_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    parts = card_number.replace(" ", "")
    assert len(parts) == 16

def test_credit_card_number_american_express_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    parts = card_number.replace(" ", "")
    assert len(parts) == 15


# LLM-generated content at query #6
#--------------------------

def test_credit_card_number_visa_no_while_loop():
    payment = Payment(seed=12345)
    result = payment.credit_card_number(card_type=CardType.VISA)
    number_part = result.replace(" ", "")
    assert len(number_part) == 16
    assert number_part.startswith("4")

def test_credit_card_number_mastercard_no_while_loop():
    payment = Payment(seed=12345)
    result = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    number_part = result.replace(" ", "")
    assert len(number_part) == 16
    assert number_part.startswith(("2", "5"))

def test_credit_card_number_american_express_no_while_loop():
    payment = Payment(seed=12345)
    result = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    number_part = result.replace(" ", "")
    assert len(number_part) == 15
    assert number_part.startswith(("34", "37"))


# LLM-generated content at query #7
#--------------------------

def test_credit_card_number_visa_no_while_loop():
    payment = Payment(seed=42)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    number = int(card_number.replace(" ", ""))
    assert 4000000000000000 <= number <= 4999999999999999
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_mastercard_no_while_loop():
    payment = Payment(seed=42)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    number = int(card_number.replace(" ", "")[:4])
    assert (2221 <= number <= 2720) or (5100 <= number <= 5599)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_american_express_no_while_loop():
    payment = Payment(seed=42)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    number = int(card_number.replace(" ", "")[:2])
    assert number == 34 or number == 37
    assert len(card_number.replace(" ", "")) == 15


# LLM-generated content at query #8
#--------------------------

def test_credit_card_number_visa_no_while_loop():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert card_number is not None
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_mastercard_no_while_loop():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert card_number is not None
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith(("2", "5"))

def test_credit_card_number_american_express_no_while_loop():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert card_number is not None
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith(("34", "37"))


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
    assert (first_four >= 2221 and first_four <= 2720) or (first_two >= 51 and first_two <= 55)

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

def test_credit_card_number_luhn_valid():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    digits = result.replace(" ", "")
    check = 0
    for i, s in enumerate(reversed(digits)):
        sx = int(s)
        if i % 2 == 1:
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
    from mimesis.enums import CardType
    from mimesis.providers.payment import Payment
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert card_number is not None
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_mastercard_length_condition_false():
    from mimesis.enums import CardType
    from mimesis.providers.payment import Payment
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert card_number is not None
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_american_express_length_condition_false():
    from mimesis.enums import CardType
    from mimesis.providers.payment import Payment
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert card_number is not None
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 15


# LLM-generated content at query #3
#--------------------------

def test_credit_card_number_visa():
    payment = Payment()
    number = payment.credit_card_number(CardType.VISA)
    parts = number.split()
    assert len(parts) == 4
    assert all(len(part) == 4 for part in parts)
    assert parts[0].startswith('4')
    digits = ''.join(parts)
    check_digit = int(digits[-1])
    total = 0
    for i, d in enumerate(reversed(digits[:-1])):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    assert (total * 9) % 10 == check_digit

def test_credit_card_number_master_card():
    payment = Payment()
    number = payment.credit_card_number(CardType.MASTER_CARD)
    parts = number.split()
    assert len(parts) == 4
    assert all(len(part) == 4 for part in parts)
    first_four = int(parts[0])
    assert (2221 <= first_four <= 2720) or (5100 <= first_four <= 5599)
    digits = ''.join(parts)
    check_digit = int(digits[-1])
    total = 0
    for i, d in enumerate(reversed(digits[:-1])):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    assert (total * 9) % 10 == check_digit

def test_credit_card_number_american_express():
    payment = Payment()
    number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    parts = number.split()
    assert len(parts) == 3
    assert len(parts[0]) == 4
    assert len(parts[1]) == 6
    assert len(parts[2]) == 5
    first_two = int(parts[0][:2])
    assert first_two in [34, 37]
    digits = ''.join(parts)
    check_digit = int(digits[-1])
    total = 0
    for i, d in enumerate(reversed(digits[:-1])):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    assert (total * 9) % 10 == check_digit

def test_credit_card_number_default_random():
    payment = Payment()
    number = payment.credit_card_number()
    parts = number.split()
    assert len(parts) in [3, 4]
    digits = ''.join(parts)
    check_digit = int(digits[-1])
    total = 0
    for i, d in enumerate(reversed(digits[:-1])):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    assert (total * 9) % 10 == check_digit

def test_credit_card_number_unsupported_type():
    payment = Payment()
    try:
        payment.credit_card_number("unsupported")
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #4
#--------------------------

def test_credit_card_number_visa_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    number_without_spaces = card_number.replace(" ", "")
    assert len(number_without_spaces) == 16

def test_credit_card_number_mastercard_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    number_without_spaces = card_number.replace(" ", "")
    assert len(number_without_spaces) == 16

def test_credit_card_number_american_express_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    number_without_spaces = card_number.replace(" ", "")
    assert len(number_without_spaces) == 15


# LLM-generated content at query #5
#--------------------------

def test_credit_card_number_visa_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    groups = card_number.split()
    assert len(groups) == 4
    assert all(len(group) == 4 for group in groups)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_mastercard_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    groups = card_number.split()
    assert len(groups) == 4
    assert all(len(group) == 4 for group in groups)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_american_express_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    groups = card_number.split()
    assert len(groups) == 3
    assert len(groups[0]) == 4
    assert len(groups[1]) == 6
    assert len(groups[2]) == 5
    assert len(card_number.replace(" ", "")) == 15


# LLM-generated content at query #6
#--------------------------

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    assert result is not None
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    result = payment.credit_card_number(CardType.MASTER_CARD)
    assert result is not None
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    first_four = int(result[:4])
    assert (2221 <= first_four <= 2720) or (5100 <= first_four <= 5599)

def test_credit_card_number_american_express():
    payment = Payment()
    result = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert result is not None
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 15
    first_two = int(result[:2])
    assert first_two in [34, 37]

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


# LLM-generated content at query #7
#--------------------------

def test_credit_card_number_visa_no_while_loop():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_mastercard_no_while_loop():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith(("2", "5"))

def test_credit_card_number_american_express_no_while_loop():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith(("34", "37"))


# LLM-generated content at query #8
#--------------------------

def test_credit_card_number_visa_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    groups = card_number.split()
    assert len(groups) == 4
    assert all(len(group) == 4 for group in groups)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_master_card_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    groups = card_number.split()
    assert len(groups) == 4
    assert all(len(group) == 4 for group in groups)
    assert len(card_number.replace(" ", "")) == 16

def test_credit_card_number_american_express_length():
    payment = Payment(seed=12345)
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    groups = card_number.split()
    assert len(groups) == 3
    assert len(groups[0]) == 4
    assert len(groups[1]) == 6
    assert len(groups[2]) == 5
    assert len(card_number.replace(" ", "")) == 15


