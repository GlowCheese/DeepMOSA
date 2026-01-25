####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    result = payment.credit_card_number()
    assert isinstance(result, str)
    assert len(result) in (19, 22)  # 16-digit or 15-digit with spaces
    assert result.count(' ') in (3, 4)  # 4 groups for 16-digit, 3 groups for 15-digit

def test_credit_card_number_visa():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.VISA)
    assert result.startswith('4')
    assert len(result) == 19  # 16-digit with spaces

def test_credit_card_number_mastercard():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert result.startswith(('2', '5'))
    assert len(result) == 19  # 16-digit with spaces

def test_credit_card_number_amex():
    payment = Payment()
    result = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert result.startswith(('34', '37'))
    assert len(result) == 22  # 15-digit with spaces

def test_credit_card_number_luhn_valid():
    payment = Payment()
    result = payment.credit_card_number()
    num = result.replace(' ', '')
    assert luhn_checksum(num[:-1]) == num[-1]

def test_credit_card_number_invalid_card_type():
    payment = Payment()
    try:
        payment.credit_card_number(card_type=CardType.DISCOVER)
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #2
#--------------------------

```python
def test_credit_card_number_length_predicate():
    payment = Payment()
    str_num = "34"
    length = 15
    assert not (len(str_num) < length - 1)


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
    assert result.startswith("2") or result.startswith("5")

def test_credit_card_number_american_express():
    payment = Payment()
    result = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert isinstance(result, str)
    assert len(result.replace(" ", "")) == 15
    assert result.startswith("3")

def test_credit_card_number_invalid_card_type():
    payment = Payment()
    try:
        payment.credit_card_number(CardType.DISCOVER)
        assert False, "Expected NotImplementedError"
    except NonEnumerableError:
        pass


# LLM-generated content at query #4
#--------------------------

```python
def test_credit_card_number_length_predicate():
    payment = Payment()
    str_num = "34"
    length = 15
    assert not (len(str_num) < length - 1)


# LLM-generated content at query #5
#--------------------------

```python
def test_credit_card_number_length_predicate():
    payment = Payment()
    card_type = CardType.AMERICAN_EXPRESS
    length = 15
    number = 34
    str_num = str(number)
    assert not (len(str_num) < length - 1)


# LLM-generated content at query #6
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

def test_credit_card_number_luhn_valid():
    payment = Payment()
    for _ in range(10):
        card_number = payment.credit_card_number().replace(" ", "")
        checksum = luhn_checksum(card_number[:-1])
        assert card_number.endswith(checksum)


# LLM-generated content at query #7
#--------------------------

```python
def test_credit_card_number_length_predicate():
    payment = Payment()
    card_type = CardType.AMERICAN_EXPRESS
    number = 34
    length = 15
    str_num = str(number)
    assert not (len(str_num) < length - 1)


# LLM-generated content at query #8
#--------------------------

```python
def test_credit_card_number_visa():
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_mastercard():
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith(("2", "5"))

def test_credit_card_number_american_express():
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("3")

def test_credit_card_number_default():
    payment = Payment()
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) in {15, 16}

def test_credit_card_number_luhn_checksum():
    payment = Payment()
    card_number = payment.credit_card_number().replace(" ", "")
    assert int(card_number[-1]) == int(luhn_checksum(card_number[:-1]))


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_credit_card_number_default():
    payment = Payment()
    card_number = payment.credit_card_number()
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_visa():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("4")

def test_credit_card_number_master_card():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.MASTER_CARD)
    assert len(card_number.replace(" ", "")) == 16
    assert card_number.startswith("2") or card_number.startswith("5")

def test_credit_card_number_american_express():
    payment = Payment()
    card_number = payment.credit_card_number(CardType.AMERICAN_EXPRESS)
    assert len(card_number.replace(" ", "")) == 15
    assert card_number.startswith("34") or card_number.startswith("37")

def test_credit_card_number_luhn_checksum():
    payment = Payment()
    card_number = payment.credit_card_number().replace(" ", "")
    assert luhn_checksum(card_number[:-1]) == card_number[-1]


# LLM-generated content at query #2
#--------------------------

```python
def test_credit_card_number_length_check():
    payment = Payment()
    card_number = payment.credit_card_number(card_type=CardType.VISA)
    assert len(card_number.replace(" ", "")) == 16


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
    card_type = CardType.AMERICAN_EXPRESS
    length = 15
    number = 34
    str_num = str(number)
    assert not (len(str_num) < length - 1)


# LLM-generated content at query #5
#--------------------------

```python
def test_credit_card_number_length_predicate():
    payment = Payment()
    card_type = CardType.AMERICAN_EXPRESS
    length = 15
    number = 34
    str_num = str(number)
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
    assert not len(str_num) < length - 1


# LLM-generated content at query #7
#--------------------------

```python
def test_credit_card_number_length_predicate():
    payment = Payment()
    card_type = CardType.AMERICAN_EXPRESS
    number = 34
    length = 15
    str_num = str(number)
    assert not (len(str_num) < length - 1)


# LLM-generated content at query #8
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

def test_credit_card_number_american_express():
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

def test_credit_card_number_invalid_card_type():
    payment = Payment()
    try:
        payment.credit_card_number(CardType.DISCOVER)
        assert False, "Expected NonEnumerableError"
    except NonEnumerableError:
        pass


