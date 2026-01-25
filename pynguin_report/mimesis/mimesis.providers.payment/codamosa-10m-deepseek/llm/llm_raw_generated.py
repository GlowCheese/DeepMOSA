####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    """Unit test for method credit_card_number of class Payment."""
    payment = Payment()
    card_types = [card_type.value for card_type in CardType]
    for card_type in card_types:
        card_type_enum = CardType(card_type)
        result = payment.credit_card_number(card_type_enum)
        assert isinstance(result, str)
        assert len(result.replace(" ", "")) in [15, 16]
        assert result.replace(" ", "").isdigit()
    assert isinstance(payment.credit_card_number(), str)
    assert len(payment.credit_card_number().replace(" ", "")) == 16
    assert payment.credit_card_number().replace(" ", "").isdigit()


# LLM-generated content at query #2
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")
    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith(("22", "23", "24", "25", "26", "27", "51", "52", "53", "54", "55"))
    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith(("34", "37"))
    # Test with invalid card type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #3
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    
    # Test with default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number) == 19  # 16 digits + 3 spaces
    assert card_number.replace(" ", "").isdigit()
    
    # Test with MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number) == 19  # 16 digits + 3 spaces
    assert card_number.replace(" ", "").isdigit()
    
    # Test with American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number) == 17  # 15 digits + 2 spaces
    assert card_number.replace(" ", "").isdigit()
    
    # Test with unsupported card type
    try:
        card_number = payment.credit_card_number("unsupported_type")
    except NonEnumerableError:
        pass
    else:
        assert False, "Expected NonEnumerableError for unsupported card type"


# LLM-generated content at query #4
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(" ", "")) == 16
    assert visa_number.startswith("4")
    
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(" ", "")) == 16
    assert mastercard_number.startswith(("2", "5"))
    
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(" ", "")) == 15
    assert amex_number.startswith(("34", "37"))
    
    # Test with invalid card type
    try:
        payment.credit_card_number("InvalidType")  # type: ignore
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #5
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    """Unit test for method credit_card_number of class Payment."""
    payment = Payment()
    visa_card = payment.credit_card_number(CardType.VISA)
    assert len(visa_card.replace(" ", "")) == 16
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    try:
        payment.credit_card_number("Unknown")
        assert False
    except NonEnumerableError:
        assert True

test_Payment_credit_card_number()


# LLM-generated content at query #6
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    
    # Test with default card_type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(" ", "")) == 16
    assert visa_number.startswith(('4'))
    
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(" ", "")) == 16
    assert mastercard_number.startswith(('2', '5'))
    
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(" ", "")) == 15
    assert amex_number.startswith(('34', '37'))
    
    # Test with invalid card_type
    try:
        payment.credit_card_number("InvalidType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #7
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    # Setup
    payment = Payment()

    # Exercise
    card_number = payment.credit_card_number()

    # Verify
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16  # Visa card number length

    # Exercise with specific card type
    card_number_mastercard = payment.credit_card_number(CardType.MASTER_CARD)

    # Verify
    assert isinstance(card_number_mastercard, str)
    assert len(card_number_mastercard.replace(" ", "")) == 16  # Mastercard card number length

    # Exercise with specific card type
    card_number_amex = payment.credit_card_number(CardType.AMERICAN_EXPRESS)

    # Verify
    assert isinstance(card_number_amex, str)
    assert len(card_number_amex.replace(" ", "")) == 15  # American Express card number length

    # Exercise with invalid card type
    try:
        payment.credit_card_number("InvalidType")
    except NonEnumerableError:
        pass
    else:
        assert False, "Expected NonEnumerableError for invalid card type"


# LLM-generated content at query #8
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    # Arrange
    payment = Payment()

    # Act
    visa_card = payment.credit_card_number(CardType.VISA)
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)

    # Assert
    assert len(visa_card.replace(" ", "")) == 16
    assert len(master_card.replace(" ", "")) == 16
    assert len(amex_card.replace(" ", "")) == 15
    assert visa_card.startswith("4")
    assert master_card.startswith(("2", "5"))
    assert amex_card.startswith(("34", "37"))


# LLM-generated content at query #9
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    # Test with MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", ")) in [16]
    # Test with AmericanExpress
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    # Test with unsupported card type
    try:
        payment.credit_card_number("unsupported_type")
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #10
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    """Test the credit_card_number method of the Payment class."""
    payment = Payment(seed=12345)
    
    # Test with default card type
    card_number = payment.credit_card_number()
    assert len(card_number) == 19  # Default Visa card format
    
    # Test with MasterCard type
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number) == 19  # MasterCard format
    
    # Test with American Express type
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number) == 17  # American Express format
    
    # Test with unsupported card type
    try:
        payment.credit_card_number("UnknownType")
    except NonEnumerableError:
        pass
    else:
        assert False, "Expected NonEnumerableError for unsupported card type"


# LLM-generated content at query #11
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card_type (should be Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith('4')
    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) in [16, 19]  # MasterCard can be 16 or 19 digits
    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith('34') or amex_card.startswith('37')
    # Test with invalid card_type (should raise NonEnumerableError)
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #12
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    # Create instance of Payment class
    payment = Payment()

    # Test generating Visa credit card number
    visa_card = payment.credit_card_number(CardType.VISA)
    assert len(visa_card) == 19  # Visa card numbers are 16 digits, formatted with spaces

    # Test generating MasterCard credit card number
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card) == 19  # MasterCard card numbers are 16 digits, formatted with spaces

    # Test generating American Express credit card number
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card) == 17  # American Express card numbers are 15 digits, formatted with spaces

    # Test generating credit card number without specifying card type
    default_card = payment.credit_card_number()
    assert len(default_card) == 19  # Default is Visa, which is 16 digits, formatted with spaces

    # Test raising NonEnumerableError when unsupported card type is provided
    try:
        payment.credit_card_number("Unsupported_Card_Type")
        assert False, "Expected NonEnumerableError not raised"
    except NonEnumerableError:
        assert True

    print("All tests passed for method credit_card_number of class Payment.")

test_Payment_credit_card_number()


# LLM-generated content at query #13
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    """Unit test for method credit_card_number of class Payment."""
    payment = Payment(seed=42)
    # Test Visa card type
    visa_card = payment.credit_card_number(CardType.VISA)
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")
    # Test MasterCard card type
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith(("22", "23", "24", "25", "26", "27", "51", "52", "53", "54", "55"))
    # Test American Express card type
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith(("34", "37"))
    # Test default card type
    default_card = payment.credit_card_number()
    assert len(default_card.replace(" ", "")) == 16
    assert default_card.startswith("4")
    # Test unsupported card type
    try:
        payment.credit_card_number("unsupported_type")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #14
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card_type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(" ", "")) == 16
    assert visa_number.startswith("4")
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(" ", "")) == 16
    assert mastercard_number.startswith(("2", "5"))
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(" ", "")) == 15
    assert amex_number.startswith(("34", "37"))
    # Test with invalid card_type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #15
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    """Test function for method credit_card_number of class Payment."""
    payment = Payment()
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) in [15, 16]

    card_number = payment.credit_card_number(CardType.VISA)
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith(("51", "52", "53", "54", "55", "2221", "2222", "2223", "2224"))

    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith(("34", "37"))


# LLM-generated content at query #16
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    """Unit test for method credit_card_number of class Payment."""
    payment = Payment()

    # Test Visa card
    visa_card = payment.credit_card_number(CardType.VISA)
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")

    # Test MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith(("5", "2"))

    # Test American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith(("34", "37"))

    # Test default card type
    default_card = payment.credit_card_number()
    assert len(default_card.replace(" ", "")) == 16
    assert default_card.startswith("4")

    # Test non-enumerable card type
    try:
        payment.credit_card_number("InvalidCardType")
    except NonEnumerableError:
        assert True
    else:
        assert False


# LLM-generated content at query #17
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(" ", "")) == 16
    assert visa_number.startswith("4")
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(" ", "")) == 16
    assert mastercard_number.startswith(("22", "23", "24", "25", "26", "27", "51", "52", "53", "54", "55"))
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(" ", "")) == 15
    assert amex_number.startswith(("34", "37"))
    # Test with invalid card type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #18
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():  # noqa: N802
    payment = Payment()
    card_types = [CardType.VISA, CardType.MASTER_CARD, CardType.AMERICAN_EXPRESS]

    for card_type in card_types:
        card_number = payment.credit_card_number(card_type)
        assert isinstance(card_number, str)
        assert len(card_number.replace(" ", "")) in [15, 16]

    # Test with None card_type
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16

    # Test with invalid card_type
    try:
        payment.credit_card_number("invalid_card_type")  # type: ignore
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #19
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment(seed=42)
    assert payment.credit_card_number(CardType.VISA) == '4455 5299 1152 2450'
    assert payment.credit_card_number(CardType.MASTER_CARD) == '2221 0000 0000 0009'
    assert payment.credit_card_number(CardType.AMERICAN_EXPRESS) == '3710 000000 00005'


# LLM-generated content at query #20
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    # Test with MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    # Test with American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    # Test with invalid card type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False
    except NonEnumerableError:
        assert True


# LLM-generated content at query #21
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():  # noqa: N802
    payment = Payment()
    # Test with default card_type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")

    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith(("22", "51", "52", "53", "54", "55"))

    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith(("34", "37"))

    # Test with invalid card_type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass

    # Test Luhn checksum
    def luhn_check(card_number: str) -> bool:
        digits = [int(d) for d in card_number.replace(" ", "")]
        checksum = sum(digits[-1::-2])
        for d in digits[-2::-2]:
            checksum += sum(divmod(d * 2, 10))
        return checksum % 10 == 0

    assert luhn_check(visa_card)
    assert luhn_check(master_card)
    assert luhn_check(amex_card)


# LLM-generated content at query #22
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    # Create an instance of Payment
    payment = Payment()

    # Test with Visa card type
    visa_number = payment.credit_card_number(CardType.VISA)
    assert re.match(r"4\d{3} \d{4} \d{4} \d{4}", visa_number), "Visa card number format is incorrect"

    # Test with MasterCard card type
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"(222[1-9]|22[3-9]\d|2[3-9]\d{2}|[3-9]\d{3}|5[1-5]\d{2}) \d{4} \d{4} \d{4}", mastercard_number), "MasterCard card number format is incorrect"

    # Test with American Express card type
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"(34|37) \d{6} \d{5}", amex_number), "American Express card number format is incorrect"

    # Test with default card type (should be Visa)
    default_number = payment.credit_card_number()
    assert re.match(r"4\d{3} \d{4} \d{4} \d{4}", default_number), "Default card number format is incorrect"

    # Test with an unsupported card type
    try:
        payment.credit_card_number("UnsupportedType")
    except NonEnumerableError:
        pass
    else:
        assert False, "Expected NonEnumerableError for unsupported card type"


# LLM-generated content at query #23
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    card_type = CardType.VISA
    result = payment.credit_card_number(card_type)
    assert isinstance(result, str)
    assert len(result.replace(' ', '')) == 16
    assert result.startswith('4')

    card_type = CardType.MASTER_CARD
    result = payment.credit_card_number(card_type)
    assert isinstance(result, str)
    assert len(result.replace(' ', '')) == 16
    assert result.startswith('5') or result.startswith('2')

    card_type = CardType.AMERICAN_EXPRESS
    result = payment.credit_card_number(card_type)
    assert isinstance(result, str)
    assert len(result.replace(' ', '')) == 15
    assert result.startswith('34') or result.startswith('37')

    try:
        card_type = 'InvalidCardType'
        payment.credit_card_number(card_type)
    except NonEnumerableError:
        assert True
    else:
        assert False


# LLM-generated content at query #24
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(" ", "")) == 16
    assert visa_number.startswith("4")
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(" ", "")) == 16
    assert mastercard_number.startswith(("2", "5"))
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(" ", "")) == 15
    assert amex_number.startswith(("34", "37"))
    # Test with invalid card type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #25
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")
    # Test MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith("5") or master_card.startswith("2")
    # Test American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith("34") or amex_card.startswith("37")
    # Test unsupported card type
    try:
        payment.credit_card_number("UnknownType")  # type: ignore
    except NonEnumerableError:
        pass
    else:
        assert False, "NonEnumerableError not raised"


# LLM-generated content at query #26
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():  # noqa: N802
    payment = Payment()
    # Test with default card_type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")

    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith(("22", "51", "52", "53", "54", "55"))

    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith(("34", "37"))

    # Test with invalid card_type
    try:
        payment.credit_card_number("InvalidCardType")  # type: ignore
    except NonEnumerableError:
        pass
    else:
        assert False, "NonEnumerableError not raised"


# LLM-generated content at query #27
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(" ", "")) == 16
    assert visa_number.startswith("4")
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(" ", "")) == 16
    assert mastercard_number.startswith(("2", "5"))
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(" ", "")) == 15
    assert amex_number.startswith(("34", "37"))
    # Test with invalid card type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #28
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    # Setup
    payment = Payment()
    card_type = CardType.VISA

    # Exercise
    result = payment.credit_card_number(card_type)
    
    # Verify
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) >= 15
    assert len(result.replace(" ", "")) <= 16



# LLM-generated content at query #29
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    card_type = CardType.MASTER_CARD
    card_number = payment.credit_card_number(card_type)
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith(('2221', '2222', '2720', '5100', '5599'))


# LLM-generated content at query #30
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number(): 
    payment = Payment() 
    
    # Test Visa card type 
    visa_number = payment.credit_card_number(CardType.VISA) 
    assert len(visa_number.replace(" ", "")) == 16 
    assert visa_number.startswith("4") 
    
    # Test MasterCard card type 
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD) 
    assert len(mastercard_number.replace(" ", "")) == 16 
    assert mastercard_number.startswith("22") or mastercard_number.startswith("5") 
    
    # Test American Express card type 
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS) 
    assert len(amex_number.replace(" ", "")) == 15 
    assert amex_number.startswith("34") or amex_number.startswith("37") 
    
    # Test invalid card type 
    try: 
        payment.credit_card_number("InvalidCardType") 
    except NonEnumerableError: 
        pass 
    else: 
        assert False, "Expected NonEnumerableError for invalid card type"


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    card_number = payment.credit_card_number()
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16
    # Test MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 16
    # Test American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert isinstance(card_number, str)
    assert len(card_number.replace(" ", "")) == 15


# LLM-generated content at query #2
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():  # noqa: N802
    """Test method credit_card_number of class Payment."""
    payment = Payment()

    # Test with default card_type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")

    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith(("5", "2"))

    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith(("34", "37"))

    # Test with invalid card_type
    try:
        payment.credit_card_number("InvalidType")  # type: ignore
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass

    # Test Luhn checksum validation
    # Visa
    visa_card = payment.credit_card_number(CardType.VISA)
    digits = [int(c) for c in visa_card.replace(" ", "")]
    checksum = 0
    for i, digit in enumerate(digits[:-1]):
        if i % 2 == 0:
            doubled = digit * 2
            checksum += doubled if doubled < 10 else doubled - 9
        else:
            checksum += digit
    assert (checksum * 9) % 10 == digits[-1]

    # MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    digits = [int(c) for c in master_card.replace(" ", "")]
    checksum = 0
    for i, digit in enumerate(digits[:-1]):
        if i % 2 == 0:
            doubled = digit * 2
            checksum += doubled if doubled < 10 else doubled - 9
        else:
            checksum += digit
    assert (checksum * 9) % 10 == digits[-1]

    # American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    digits = [int(c) for c in amex_card.replace(" ", "")]
    checksum = 0
    for i, digit in enumerate(digits[:-1]):
        if i % 2 == 0:
            doubled = digit * 2
            checksum += doubled if doubled < 10 else doubled - 9
        else:
            checksum += digit
    assert (checksum * 9) % 10 == digits[-1]


# LLM-generated content at query #3
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    visa_number = payment.credit_card_number(CardType.VISA)
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)

    assert len(visa_number) == 19
    assert len(mastercard_number) == 19
    assert len(amex_number) == 17

    assert visa_number.startswith('4')
    assert mastercard_number.startswith('2') or mastercard_number.startswith('5')
    assert amex_number.startswith('34') or amex_number.startswith('37')

    assert re.match(r'^\d{4} \d{4} \d{4} \d{4}$', visa_number)
    assert re.match(r'^\d{4} \d{4} \d{4} \d{4}$', mastercard_number)
    assert re.match(r'^\d{4} \d{6} \d{5}$', amex_number)


# LLM-generated content at query #4
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card_type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(" ", "")) == 16
    assert visa_number.startswith('4')
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(" ", "")) == 16
    assert mastercard_number.startswith('5') or mastercard_number.startswith('2')
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(" ", "")) == 15
    assert amex_number.startswith('34') or amex_number.startswith('37')
    # Test with invalid card_type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #5
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(" ", "")) == 16
    assert visa_number.startswith("4")
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(" ", "")) == 16
    assert mastercard_number.startswith(("2", "5"))
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(" ", "")) == 15
    assert amex_number.startswith(("34", "37"))
    # Test with invalid card type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        assert True


# LLM-generated content at query #6
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    card_type = CardType.VISA
    card_number = payment.credit_card_number(card_type)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")
    card_type = CardType.MASTER_CARD
    card_number = payment.credit_card_number(card_type)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("22") or card_number.startswith("5")
    card_type = CardType.AMERICAN_EXPRESS
    card_number = payment.credit_card_number(card_type)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")



# LLM-generated content at query #7
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    
    # Test with default card_type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(" ", "")) == 16
    assert visa_number.startswith("4")
    
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(" ", "")) == 16
    assert mastercard_number.startswith(("2", "5"))
    
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(" ", "")) == 15
    assert amex_number.startswith(("34", "37"))
    
    # Test with invalid card_type
    try:
        payment.credit_card_number("InvalidCardType")  # type: ignore
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #8
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test Visa card number generation
    visa_card = payment.credit_card_number(CardType.VISA)
    assert re.match(r'^4\d{3} \d{4} \d{4} \d{4}$', visa_card) is not None
    # Test MasterCard card number generation
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r'^(2221|2720|5[1-5]\d{2}) \d{4} \d{4} \d{4}$', master_card) is not None
    # Test American Express card number generation
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r'^(34|37)\d{2} \d{6} \d{5}$', amex_card) is not None
    # Test invalid card type
    try:
        payment.credit_card_number("InvalidType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass



# LLM-generated content at query #9
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith('4')
    # Test with MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith('5') or card_number.startswith('2')
    # Test with American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith('34') or card_number.startswith('37')


# LLM-generated content at query #10
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test Visa card
    visa_card = payment.credit_card_number(CardType.VISA)
    assert re.match(r"4\d{3} \d{4} \d{4} \d{4}", visa_card)
    
    # Test MasterCard card
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"(2221 \d{4} \d{4} \d{4})|(2720 \d{4} \d{4} \d{4})|(51\d{2} \d{4} \d{4} \d{4})|(55\d{2} \d{4} \d{4} \d{4})", master_card)
    
    # Test American Express card
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"(34|37)\d{2} \d{6} \d{5}", amex_card)
    
    # Test invalid card type
    try:
        payment.credit_card_number("InvalidCardType")
    except NonEnumerableError:
        pass
    else:
        assert False, "Expected NonEnumerableError for invalid card type"


# LLM-generated content at query #11
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    """Test method credit_card_number of class Payment."""
    payment = Payment()
    # Test with default card_type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(" ", "")) == 16
    assert visa_number.startswith(("4", "5", "2"))
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(" ", "")) == 16
    assert mastercard_number.startswith(("5", "2"))
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(" ", "")) == 15
    assert amex_number.startswith(("34", "37"))
    # Test with invalid card_type
    try:
        payment.credit_card_number("InvalidCardType")  # type: ignore
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #12
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card_type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(" ", "")) == 16
    assert visa_number.startswith("4")
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(" ", "")) == 16
    assert mastercard_number.startswith(("2", "5"))
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(" ", "")) == 15
    assert amex_number.startswith(("34", "37"))
    # Test with invalid card_type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #13
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card_type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith('4')
    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith('5') or master_card.startswith('2')
    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith('34') or amex_card.startswith('37')
    # Test with invalid card_type
    try:
        payment.credit_card_number("InvalidType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #14
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith(('4'))
    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith(('2', '5'))
    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith(('34', '37'))
    # Test with invalid card type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #15
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    assert len(payment.credit_card_number()) == 19
    assert len(payment.credit_card_number(CardType.MASTER_CARD)) == 19
    assert len(payment.credit_card_number(CardType.AMERICAN_EXPRESS)) == 17


# LLM-generated content at query #16
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")
    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith("5") or master_card.startswith("2")
    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith("34") or amex_card.startswith("37")
    # Test with unsupported card type
    try:
        payment.credit_card_number("UnsupportedCardType")
    except NonEnumerableError:
        assert True
    else:
        assert False


# LLM-generated content at query #17
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    
    # Test default card type (Visa)
    visa_card = payment.credit_card_number()
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", visa_card) is not None
    assert visa_card.startswith('4')
    
    # Test MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert re.match(r"^\d{4} \d{4} \d{4} \d{4}$", master_card) is not None
    assert master_card.startswith('2221') or master_card.startswith('2720') or master_card.startswith('5100') or master_card.startswith('5599')
    
    # Test American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert re.match(r"^\d{4} \d{6} \d{5}$", amex_card) is not None
    assert amex_card.startswith('34') or amex_card.startswith('37')
    
    # Test unsupported card type
    try:
        payment.credit_card_number(CardType(999))
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #18
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test Visa
    visa = payment.credit_card_number(CardType.VISA)
    assert len(visa.replace(" ", "")) == 16
    assert visa.startswith("4")
    # Test MasterCard
    mastercard = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard.replace(" ", "")) == 16
    assert mastercard.startswith("222") or mastercard.startswith("272") or mastercard.startswith("51") or mastercard.startswith("55")
    # Test American Express
    amex = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex.replace(" ", "")) == 15
    assert amex.startswith("34") or amex.startswith("37")
    # Test NonEnumerableError
    try:
        payment.credit_card_number("InvalidType")
    except NonEnumerableError:
        pass
    else:
        assert False, "Expected NonEnumerableError"



# LLM-generated content at query #19
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    
    # Test for Visa card type
    visa_card = payment.credit_card_number(CardType.VISA)
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith('4')
    
    # Test for MasterCard card type
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith('22') or master_card.startswith('51') or master_card.startswith('55')
    
    # Test for American Express card type
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith('34') or amex_card.startswith('37')
    
    # Test for default card type (Visa)
    default_card = payment.credit_card_number()
    assert len(default_card.replace(" ", "")) == 16
    assert default_card.startswith('4')
    
    # Test for NonEnumerableError
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        assert True


# LLM-generated content at query #20
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    assert payment.credit_card_number(CardType.VISA)
    assert payment.credit_card_number(CardType.MASTER_CARD)
    assert payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert payment.credit_card_number()



# LLM-generated content at query #21
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")

    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith(("5", "2"))

    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith(("34", "37"))

    # Test with invalid card type
    try:
        payment.credit_card_number("InvalidType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #22
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    """Unit test for method credit_card_number of class Payment."""
    payment = Payment()
    # Test Visa
    visa_card = payment.credit_card_number(CardType.VISA)
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")
    # Test MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith("5") or master_card.startswith("2")
    # Test American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith("34") or amex_card.startswith("37")
    # Test default card type (Visa)
    default_card = payment.credit_card_number()
    assert len(default_card.replace(" ", "")) == 16
    assert default_card.startswith("4")


# LLM-generated content at query #23
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    """Unit test for method credit_card_number of class Payment."""
    payment = Payment()
    
    # Test Visa card
    visa_card = payment.credit_card_number(CardType.VISA)
    assert len(visa_card) == 19  # Includes spaces
    assert visa_card.startswith('4')
    
    # Test MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card) == 19  # Includes spaces
    assert master_card.startswith('5') or master_card.startswith('2')
    
    # Test American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card) == 17  # Includes spaces
    assert amex_card.startswith('34') or amex_card.startswith('37')
    
    # Test default card type (Visa)
    default_card = payment.credit_card_number()
    assert len(default_card) == 19  # Includes spaces
    assert default_card.startswith('4')
    
    # Test invalid card type
    try:
        payment.credit_card_number('InvalidType')
    except NonEnumerableError:
        pass
    else:
        assert False, "Expected NonEnumerableError"


# LLM-generated content at query #24
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test Visa card type
    visa_card = payment.credit_card_number(CardType.VISA)
    assert len(visa_card) == 19
    assert visa_card.startswith('4') or visa_card.startswith('4 4')
    # Test MasterCard card type
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card) == 19
    assert master_card.startswith('2221') or master_card.startswith('2720') or master_card.startswith('5100') or master_card.startswith('5599') or master_card.startswith('2221 ') or master_card.startswith('2720 ') or master_card.startswith('5100 ') or master_card.startswith('5599 ')
    # Test American Express card type
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card) == 17
    assert amex_card.startswith('34') or amex_card.startswith('37') or amex_card.startswith('34 ') or amex_card.startswith('37 ')
    # Test default card type
    default_card = payment.credit_card_number()
    assert len(default_card) == 19
    assert default_card.startswith('4') or default_card.startswith('4 4')


# LLM-generated content at query #25
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number(): 
    p = Payment()
    card_type = CardType.VISA
    result = p.credit_card_number(card_type)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith('4')
    
    card_type = CardType.MASTER_CARD
    result = p.credit_card_number(card_type)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith('2') or result.startswith('5')
    
    card_type = CardType.AMERICAN_EXPRESS
    result = p.credit_card_number(card_type)
    assert len(result.replace(" ", "")) == 15
    assert result.startswith('34') or result.startswith('37')


# LLM-generated content at query #26
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card type (Visa)
    visa_number = payment.credit_card_number()
    assert len(visa_number.replace(' ', '')) == 16
    assert visa_number.startswith('4')
    # Test with MasterCard
    mastercard_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(mastercard_number.replace(' ', '')) == 16
    assert mastercard_number.startswith(('22', '23', '24', '25', '26', '27', '51', '52', '53', '54', '55'))
    # Test with American Express
    amex_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_number.replace(' ', '')) == 15
    assert amex_number.startswith(('34', '37'))
    # Test with unsupported card type
    try:
        payment.credit_card_number('unsupported_type')
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #27
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    """Unit test for method credit_card_number of class Payment."""
    payment = Payment()
    # Test Visa
    visa_card = payment.credit_card_number(CardType.VISA)
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")
    # Test MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith("5") or master_card.startswith("2")
    # Test American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith("34") or amex_card.startswith("37")
    # Test invalid card type
    try:
        payment.credit_card_number("InvalidCardType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #28
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    """Test method credit_card_number of class Payment."""
    payment = Payment()

    # Test with default card type (Visa)
    card_number = payment.credit_card_number()
    assert len(card_number) == 19  # Visa card number length with spaces
    assert card_number.replace(" ", "").isdigit()

    # Test with MasterCard
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number) == 19  # MasterCard card number length with spaces
    assert card_number.replace(" ", "").isdigit()

    # Test with American Express
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number) == 17  # American Express card number length with spaces
    assert card_number.replace(" ", "").isdigit()

    # Test with unsupported card type (should raise NonEnumerableError)
    try:
        payment.credit_card_number("UnsupportedCardType")  # type: ignore
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass



# LLM-generated content at query #29
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    for _ in range(100):
        card_number = payment.credit_card_number()
        assert len(card_number.replace(" ", "")) == 16
        assert card_number.count(" ") == 3
        assert card_number[:4] in ["4000", "4001", "4002", "4003", "4004", "4005", "4006", "4007", "4008", "4009", "4010", "4011", "4012", "4013", "4014", "4015", "4016", "4017", "4018", "4019", "4020", "4021", "4022", "4023", "4024", "4025", "4026", "4027", "4028", "4029", "4030", "4031", "4032", "4033", "4034", "4035", "4036", "4037", "4038", "4039", "4040", "4041", "4042", "4043", "4044", "4045", "4046", "4047", "4048", "4049", "4050", "4051", "4052", "4053", "4054", "4055", "4056", "4057", "4058", "4059", "4060", "4061", "4062", "4063", "4064", "4065", "4066", "4067", "4068", "4069", "4070", "4071", "4072", "4073", "4074", "4075", "4076", "4077", "4078", "4079", "4080", "4081", "4082", "4083", "4084", "4085", "4086", "4087", "4088", "4089", "4090", "4091", "4092", "4093", "4094", "4095", "4096", "4097", "4098", "4099", "4100", "4101", "4102", "4103", "4104", "4105", "4106", "4107", "4108", "4109", "4110", "4111", "4112", "4113", "4114", "4115", "4116", "4117", "4118", "4119", "4120", "4121", "4122", "4123", "4124", "4125", "4126", "4127", "4128", "4129", "4130", "4131", "4132", "4133", "4134", "4135", "4136", "4137", "4138", "4139", "4140", "4141", "4142", "4143", "4144", "4145", "4146", "4147", "4148", "4149", "4150", "4151", "4152", "4153", "4154", "4155", "4156", "4157", "4158", "4159", "4160", "4161", "4162", "4163", "4164", "4165", "4166", "4167", "4168", "4169", "4170", "4171", "4172", "4173", "4174", "4175", "4176", "4177", "4178", "4179", "4180", "4181", "4182", "4183", "4184", "4185", "4186", "4187", "4188", "4189", "4190", "4191", "4192", "4193", "4194", "4195", "4196", "4197", "4198", "4199", "4200", "4201", "4202", "4203", "4204", "4205", "4206", "4207", "4208", "4209", "4210", "4211", "4212", "4213", "4214", "4215", "4216", "4217", "4218", "4219", "4220", "4221", "4222", "4223", "4224", "4225", "4226", "4227", "4228", "4229", "4230", "4231", "4232", "4233", "4234", "4235", "4236", "4237", "4238", "4239", "4240", "4241", "4242", "4243", "4244", "4245", "4246", "4247", "4248", "4249", "4250", "4251", "4252", "4253", "4254", "4255", "4256", "4257", "4258", "4259", "4260", "4261", "4262", "4263", "4264", "4265", "4266", "4267", "4268", "4269", "4270", "4271", "4272", "4273", "4274", "4275", "4276", "4277", "4278", "4279", "4280", "4281", "4282", "4283", "4284", "4285", "4286", "4287", "4288", "4289", "4290", "4291", "4292", "4293", "4294", "4295", "4296", "4297", "4298", "4299", "4300", "4301", "4302", "4303", "4304", "4305", "4306", "4307", "4308", "4309", "4310", "4311", "4312", "4313", "4314", "4315", "4316", "4317", "4318", "4319", "4320", "4321", "4322", "4323", "4324", "4325", "4326", "4327", "4328", "4329", "4330", "4331", "4332", "4333", "4334", "4335", "4336", "4337", "4338", "4339", "4340", "4341", "4342", "4343", "4344", "4345", "4346", "4347", "4348", "4349", "4350", "4351", "4352", "4353", "4354", "4355", "4356", "4357", "4358", "4359", "4360", "4361", "4362", "4363", "4364", "4365", "4366", "4367", "4368", "4369", "4370", "4371", "4372", "4373", "4374", "4375", "4376", "4377", "4378", "4379", "4380", "4381", "4382", "4383", "4384", "4385", "4386", "4387", "4388", "4389", "4390", "4391", "4392", "4393", "4394", "4395", "4396", "4397", "4398", "4399", "4400", "4401", "4402", "4403", "4404", "4405", "4406", "4407", "4408", "4409", "4410", "4411", "4412", "4413", "4414", "4415", "4416", "4417", "4418", "4419", "4420", "4421", "4422", "4423", "4424", "4425", "4426", "4427", "4428", "4429", "4430", "4431", "4432", "4433", "4434", "4435", "4436", "4437", "4438", "4439", "4440", "4441", "4442", "4443", "4444", "4445", "4446", "4447", "4448", "4449", "4450", "4451", "4452", "4453", "4454", "4455", "4456", "4457", "4458", "4459", "4460", "4461", "4462", "4463", "4464", "4465", "4466", "4467", "4468", "4469", "4470", "4471", "4472", "4473", "4474", "4475", "4476", "4477", "4478", "4479", "4480", "4481", "4482", "4483", "4484", "4485", "4486", "4487", "4488", "4489", "4490", "4491", "4492", "4493", "4494", "4495", "4496", "449


# LLM-generated content at query #30
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    # Arrange
    payment = Payment()

    # Act
    visa_card = payment.credit_card_number(CardType.VISA)
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)

    # Assert
    assert len(visa_card) == 19  # Including spaces
    assert len(master_card) == 19  # Including spaces
    assert len(amex_card) == 17  # Including spaces

    assert visa_card[0] == '4'
    assert master_card[:4] in ['2221', '2720', '5100', '5599']
    assert amex_card[:2] in ['34', '37']


# LLM-generated content at query #31
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():  # noqa: N802
    """Test method credit_card_number of class Payment."""
    payment = Payment()
    # Test with default card_type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")

    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith(("5", "2"))

    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith(("34", "37"))

    # Test with invalid card_type
    try:
        payment.credit_card_number("invalid_card_type")  # type: ignore
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #32
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test with default card_type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card) == 19
    assert visa_card[:1] == '4'
    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card) == 19
    assert master_card[:1] == '2' or master_card[:1] == '5'
    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card) == 17
    assert amex_card[:1] == '3'
    # Test with invalid card_type
    try:
        payment.credit_card_number('InvalidCardType')
    except NonEnumerableError:
        pass
    else:
        assert False, 'Expected NonEnumerableError'


# LLM-generated content at query #33
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number(): 
    payment = Payment()
    # Test with default card_type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")
    # Test with MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith(("5", "2"))
    # Test with American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith(("34", "37"))
    # Test with invalid card_type
    try:
        payment.credit_card_number("InvalidType")
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #34
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Test default card type (Visa)
    visa_card = payment.credit_card_number()
    assert len(visa_card.replace(" ", "")) == 16
    assert visa_card.startswith("4")
    # Test MasterCard
    master_card = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(master_card.replace(" ", "")) == 16
    assert master_card.startswith(("2221", "2720", "5100", "5599"))
    # Test American Express
    amex_card = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(amex_card.replace(" ", "")) == 15
    assert amex_card.startswith(("34", "37"))
    # Test invalid card type
    try:
        payment.credit_card_number("InvalidType")
    except NonEnumerableError:
        pass
    else:
        assert False, "Expected NonEnumerableError for invalid card type"


# LLM-generated content at query #35
#--------------------------

# Unit test for method credit_card_number of class Payment
def test_Payment_credit_card_number():
    payment = Payment()
    # Testing Visa card
    card_number = payment.credit_card_number(CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")
    # Testing MasterCard card
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("5") or card_number.startswith("2")
    # Testing American Express card
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")



