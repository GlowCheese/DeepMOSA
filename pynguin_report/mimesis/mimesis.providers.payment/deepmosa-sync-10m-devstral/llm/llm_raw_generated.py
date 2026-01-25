####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert len(result.replace(" ", "")) == 16
    assert result.count(" ") == 3

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.VISA)
    assert result.startswith("4")
    assert len(result.replace(" ", "")) == 16

def test_credit_card_number_mastercard():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert result.startswith(("2", "5"))
    assert len(result.replace(" ", "")) == 16

def test_credit_card_number_amex():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert result.startswith(("34", "37"))
    assert len(result.replace(" ", "")) == 15
    assert result.count(" ") == 2

def test_credit_card_number_luhn_checksum():
    payment = Payment()
    card_number = payment.credit_card_number()
    digits = card_number.replace(" ", "")
    checksum = luhn_checksum(digits[:-1])
    assert digits[-1] == checksum


# LLM-generated content at query #2
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.VISA)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith(("2", "5"))

def test_credit_card_number_american_express():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 15
    assert result.startswith("3")

def test_credit_card_number_luhn_checksum():
    payment = Payment()
    card_number = payment.credit_card_number()
    digits = card_number.replace(" ", "")
    checksum = luhn_checksum(digits[:-1])
    assert digits[-1] == checksum


# LLM-generated content at query #3
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    result = payment.credit_card_number(CardType.MASTER_CARD)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith(("2", "5"))

def test_credit_card_number_amex():
    payment = Payment()
    result = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 15
    assert result.startswith("3")

def test_credit_card_number_luhn_checksum():
    payment = Payment()
    card_number = payment.credit_card_number().replace(" ", "")
    checksum = luhn_checksum(card_number[:-1])
    assert card_number[-1] == checksum


# LLM-generated content at query #4
#--------------------------

```python
def test_credit_card_number_length_predicate():
    payment = Payment()
    card_type = CardType.AMERICAN_EXPRESS
    number = 34
    length = 15
    str_num = str(number)
    assert not (len(str_num) < length - 1)


# LLM-generated content at query #5
#--------------------------

```python
def test_credit_card_number_length_predicate():
    payment = Payment()
    str_num = "34"
    length = 15
    assert not (len(str_num) < length - 1)


# LLM-generated content at query #6
#--------------------------

```python
def test_credit_card_number_length_predicate():
    payment = Payment()
    card_type = CardType.AMERICAN_EXPRESS
    length = 15
    number = 34
    str_num = str(number)
    assert not (len(str_num) < length - 1)


# LLM-generated content at query #7
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.count(" ") == 3

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    assert result.startswith("4")
    assert len(result.replace(" ", "")) == 16

def test_credit_card_number_mastercard():
    payment = Payment()
    result = payment.credit_card_number(CardType.MASTER_CARD)
    assert result.startswith(("2", "5"))
    assert len(result.replace(" ", "")) == 16

def test_credit_card_number_amex():
    payment = Payment()
    result = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert result.startswith(("34", "37"))
    assert len(result.replace(" ", "")) == 15
    assert result.count(" ") == 2

def test_credit_card_number_luhn_checksum():
    payment = Payment()
    card_number = payment.credit_card_number().replace(" ", "")
    checksum = luhn_checksum(card_number[:-1])
    assert card_number.endswith(checksum)


# LLM-generated content at query #8
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    result = payment.credit_card_number(CardType.MASTER_CARD)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith(("2", "5"))

def test_credit_card_number_amex():
    payment = Payment()
    result = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 15
    assert result.startswith("3")

def test_credit_card_number_luhn_checksum():
    payment = Payment()
    card_number = payment.credit_card_number().replace(" ", "")
    checksum = luhn_checksum(card_number[:-1])
    assert card_number.endswith(checksum)


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert len(result) == 19
    assert result.count(" ") == 3
    assert result.replace(" ", "").isdigit()

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    assert result.startswith("4")
    assert len(result) == 19
    assert result.count(" ") == 3

def test_credit_card_number_mastercard():
    payment = Payment()
    result = payment.credit_card_number(CardType.MASTER_CARD)
    assert result.startswith(("2", "5"))
    assert len(result) == 19
    assert result.count(" ") == 3

def test_credit_card_number_amex():
    payment = Payment()
    result = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert result.startswith(("34", "37"))
    assert len(result) == 17
    assert result.count(" ") == 2

def test_credit_card_number_luhn_checksum():
    payment = Payment()
    card_number = payment.credit_card_number().replace(" ", "")
    checksum = luhn_checksum(card_number[:-1])
    assert card_number.endswith(checksum)


# LLM-generated content at query #2
#--------------------------

```python
def test_credit_card_number_length_condition():
    payment = Payment()
    card_type = CardType.AMERICAN_EXPRESS
    number = 34
    length = 15
    str_num = str(number)
    assert len(str_num) >= length - 1


# LLM-generated content at query #3
#--------------------------

```python
def test_credit_card_number_length_predicate():
    payment = Payment()
    card_type = CardType.AMERICAN_EXPRESS
    length = 15
    number = 34
    str_num = str(number)
    assert not (len(str_num) < length - 1)


# LLM-generated content at query #4
#--------------------------

```python
def test_credit_card_number_length_predicate():
    payment = Payment()
    assert len(payment.credit_card_number(CardType.AMERICAN_EXPRESS)) == 15


# LLM-generated content at query #5
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert len(result) == 19
    assert result.count(' ') == 3
    assert result.startswith('4')

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    assert len(result) == 19
    assert result.count(' ') == 3
    assert result.startswith('4')

def test_credit_card_number_mastercard():
    payment = Payment()
    result = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(result) == 19
    assert result.count(' ') == 3
    assert result.startswith(('2', '5'))

def test_credit_card_number_amex():
    payment = Payment()
    result = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(result) == 17
    assert result.count(' ') == 2
    assert result.startswith(('34', '37'))

def test_credit_card_number_invalid_type():
    payment = Payment()
    try:
        payment.credit_card_number(CardType.DISCOVER)
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_credit_card_number_length_check():
    payment = Payment()
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16


# LLM-generated content at query #7
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert len(result) == 19
    assert result.count(' ') == 3
    assert result.startswith('4')

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(CardType.VISA)
    assert len(result) == 19
    assert result.count(' ') == 3
    assert result.startswith('4')

def test_credit_card_number_mastercard():
    payment = Payment()
    result = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(result) == 19
    assert result.count(' ') == 3
    assert result.startswith(('2', '5'))

def test_credit_card_number_amex():
    payment = Payment()
    result = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(result) == 17
    assert result.count(' ') == 2
    assert result.startswith(('34', '37'))

def test_credit_card_number_luhn_checksum():
    payment = Payment()
    card_number = payment.credit_card_number().replace(' ', '')
    checksum = luhn_checksum(card_number[:-1])
    assert card_number.endswith(checksum)


# LLM-generated content at query #8
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.VISA)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 16
    assert result.startswith(("2", "5"))

def test_credit_card_number_amex():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 15
    assert result.startswith("3")

def test_credit_card_number_luhn_valid():
    payment = Payment()
    card_number = payment.credit_card_number().replace(" ", "")
    total = 0
    for i, digit in enumerate(reversed(card_number)):
        num = int(digit)
        if i % 2 == 0:
            num *= 2
        if num > 9:
            num -= 9
        total += num
    assert total % 10 == 0


