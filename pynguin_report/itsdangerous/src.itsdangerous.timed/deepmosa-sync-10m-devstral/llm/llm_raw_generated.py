####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data)
    assert result == "test_data"

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    invalid_data = b"invalid_data"
    try:
        serializer.loads(invalid_data)
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    try:
        serializer.loads(signed_data, max_age=-1)
        assert False, "Expected SignatureExpired exception"
    except SignatureExpired:
        pass

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == "test_data"
    assert isinstance(timestamp, int)

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data.encode("utf-8"))
    assert result == "test_data"

def test_loads_with_string_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data.decode("utf-8"))
    assert result == "test_data"

def test_loads_with_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data", salt="custom_salt")
    result = serializer.loads(signed_data, salt="custom_salt")
    assert result == "test_data"

def test_loads_unsafe_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    is_valid, result = serializer.loads_unsafe(signed_data)
    assert is_valid is True
    assert result == "test_data"

def test_loads_unsafe_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    invalid_data = b"invalid_data"
    is_valid, result = serializer.loads_unsafe(invalid_data)
    assert is_valid is False
    assert result is None

def test_loads_unsafe_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    is_valid, result = serializer.loads_unsafe(signed_data, max_age=-1)
    assert is_valid is False
    assert result is None


# LLM-generated content at query #2
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_max_age_not_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=100) == value

def test_unsign_with_valid_signature_and_max_age_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired as e:
        assert e.payload == value
    else:
        assert False, "Expected SignatureExpired"

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
    except BadSignature:
        pass
    else:
        assert False, "Expected BadSignature"

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
    except BadTimeSignature as e:
        assert e.payload == b"test"
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.sep.invalid")
    except BadTimeSignature as e:
        assert e.payload == b"test"
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired as e:
        assert e.payload == value
    else:
        assert False, "Expected SignatureExpired"


# LLM-generated content at query #3
#--------------------------

```python
def test_unsign_raises_bad_time_signature_on_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid_timestamp"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(corrupted_value)
    assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #4
#--------------------------

```python
def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2


# LLM-generated content at query #5
#--------------------------

```python
def test_unsign_with_sig_error_and_invalid_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Tamper with the timestamp to make it invalid
    tampered_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid_timestamp"
    try:
        signer.unsign(tampered_value)
    except BadTimeSignature as e:
        assert e.args[0] == "Malformed timestamp"


# LLM-generated content at query #6
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.sep.invalid_timestamp")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)


# LLM-generated content at query #7
#--------------------------

```python
def test_unsign_raises_signature_expired_when_age_exceeds_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    timestamp = int(time.time()) - 100
    signed_value = signer.sign(value)
    with patch.object(signer, "get_timestamp", return_value=timestamp + 101):
        assert raises(SignatureExpired, signer.unsign, signed_value, max_age=100)


# LLM-generated content at query #8
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.sep.malformed")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)


# LLM-generated content at query #9
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"

def test_unsign_with_valid_signature_and_max_age_not_expired():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, max_age=1000)
    assert result == b"test"

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadSignature):
        signer.unsign("invalid_signature")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.invalid_timestamp")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)


# LLM-generated content at query #10
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value"


# LLM-generated content at query #11
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data)
    assert result == "test_data"

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    try:
        serializer.loads("invalid_signature")
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    try:
        serializer.loads(signed_data, max_age=-1)
        assert False, "Expected SignatureExpired exception"
    except SignatureExpired:
        pass

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == "test_data"
    assert isinstance(timestamp, int)

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data.encode("utf-8"))
    assert result == "test_data"

def test_loads_with_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data", salt="custom_salt")
    result = serializer.loads(signed_data, salt="custom_salt")
    assert result == "test_data"


# LLM-generated content at query #12
#--------------------------

```python
def test_unsign_raises_bad_time_signature_when_timestamp_is_malformed():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


# LLM-generated content at query #13
#--------------------------

```python
def test_loads_with_return_timestamp_true():
    serializer = TimedSerializer("secret")
    payload = {"data": "test"}
    signed = serializer.dumps(payload)
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == payload
    assert isinstance(result[1], int)


# LLM-generated content at query #14
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_constructor_custom_params():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.secret_keys == [b"custom-secret"]

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #15
#--------------------------

```python
def test_signature_expired_raises_immediately():
    serializer = TimedSerializer("secret")
    s = serializer.dumps("data")
    with pytest.raises(SignatureExpired):
        serializer.loads(s, max_age=-1)


# LLM-generated content at query #16
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_max_age_not_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=1000) == value

def test_unsign_with_valid_signature_and_max_age_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired as e:
        assert e.payload == value

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
    except BadSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
    except BadTimeSignature as e:
        assert e.payload == b"test"

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.sep.invalid")
    except BadTimeSignature as e:
        assert e.payload == b"test"

def test_unsign_with_return_timestamp_true():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_return_timestamp_false():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed, return_timestamp=False)
    assert result == value
    assert not isinstance(result, tuple)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1000)
    except SignatureExpired as e:
        assert e.payload == value


# LLM-generated content at query #17
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


# LLM-generated content at query #18
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=_lazy_sha256,
    )
    assert signer.secret_keys == [b"custom-secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == _lazy_sha256()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_invalid_separator():
    with raises(ValueError):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #19
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value" + signer.sep.encode()


# LLM-generated content at query #20
#--------------------------

```python
def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2


# LLM-generated content at query #21
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


# LLM-generated content at query #22
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.secret_keys == [b"custom-secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError):
        TimestampSigner(b"secret-key", sep="a")

def test_timestamp_signer_constructor_none_salt():
    signer = TimestampSigner(b"secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #23
#--------------------------

```python
def test_unsign_raises_signature_expired_for_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    future_timestamp = signer.get_timestamp() + 100
    manipulated_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep + base64_encode(int_to_bytes(future_timestamp))
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(manipulated_value, max_age=50)
    assert exc_info.value.payload == b"value"
    assert exc_info.value.date_signed == signer.timestamp_to_datetime(future_timestamp)


# LLM-generated content at query #24
#--------------------------

```python
def test_unsign_raises_signature_expired_for_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    future_timestamp = signer.get_timestamp() + 100
    manipulated_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep + base64_encode(int_to_bytes(future_timestamp))
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(manipulated_value)
    assert "Signature age -100 < 0 seconds" in str(exc_info.value)


# LLM-generated content at query #25
#--------------------------

```python
def test_TimestampSigner_constructor_with_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_TimestampSigner_constructor_with_custom_salt():
    signer = TimestampSigner(b"secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_TimestampSigner_constructor_with_custom_sep():
    signer = TimestampSigner(b"secret-key", sep="|")
    assert signer.sep == b"|"

def test_TimestampSigner_constructor_with_custom_key_derivation():
    signer = TimestampSigner(b"secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_TimestampSigner_constructor_with_custom_digest_method():
    custom_digest = lambda: hashlib.sha256
    signer = TimestampSigner(b"secret-key", digest_method=custom_digest)
    assert signer.digest_method == custom_digest()

def test_TimestampSigner_constructor_with_custom_algorithm():
    custom_algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner(b"secret-key", algorithm=custom_algorithm)
    assert signer.algorithm == custom_algorithm

def test_TimestampSigner_constructor_with_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_TimestampSigner_constructor_with_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used because it may be contained in the signature itself"):
        TimestampSigner(b"secret-key", sep="=")

def test_TimestampSigner_constructor_with_none_salt():
    signer = TimestampSigner(b"secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #26
#--------------------------

```python
def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


# LLM-generated content at query #27
#--------------------------

```python
def test_signature_expired_exception_is_raised_immediately():
    serializer = TimedSerializer("secret")
    with pytest.raises(SignatureExpired):
        serializer.loads("expired_signature", max_age=0)


# LLM-generated content at query #28
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_parameters():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=_lazy_sha256,
        algorithm=HMACAlgorithm(_lazy_sha256)
    )
    assert signer.secret_keys == [b"custom-secret"]
    assert signer.sep == b"|"
    assert signer.salt == b"custom-salt"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == _lazy_sha256()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_string_secret_key():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_constructor_with_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner(b"secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #29
#--------------------------

```python
def test_loads_without_return_timestamp():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result = serializer.loads(signed_data)
    assert isinstance(result, str)
    assert result == "data"


# LLM-generated content at query #30
#--------------------------

```python
def test_loads_without_return_timestamp():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data)
    assert isinstance(result, str)
    assert result == "test_data"


# LLM-generated content at query #31
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value

def test_unsign_with_valid_signature_and_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    signed = b"invalid"
    try:
        signer.unsign(signed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = b"hello"
    try:
        signer.unsign(signed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = b"hello.sep.invalid"
    try:
        signer.unsign(signed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)


# LLM-generated content at query #32
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = "test_value"
    signed_value = signer.sign(value)
    result = signer.unsign(signed_value)
    assert result == b"test_value"

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = "test_value"
    signed_value = signer.sign(value)
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid_signature")
    except BadTimeSignature as e:
        assert str(e) == "Invalid signature"

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("test_value")
    except BadTimeSignature as e:
        assert str(e) == "timestamp missing"

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("test_value.sep.invalid_timestamp")
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = "test_value"
    signed_value = signer.sign(value)
    try:
        signer.unsign(signed_value, max_age=0)
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = "test_value"
    signed_value = signer.sign(value)
    try:
        signer.unsign(signed_value, max_age=-1)
    except SignatureExpired as e:
        assert "Signature age" in str(e)


# LLM-generated content at query #33
#--------------------------

```python
def test_timestamp_signer_constructor_with_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_salt():
    signer = TimestampSigner(b"secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner(b"secret-key", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_constructor_with_custom_key_derivation():
    signer = TimestampSigner(b"secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_custom_digest_method():
    custom_digest = staticmethod(lambda: hashlib.sha256)
    signer = TimestampSigner(b"secret-key", digest_method=custom_digest)
    assert signer.digest_method == custom_digest

def test_timestamp_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(staticmethod(lambda: hashlib.sha256))
    signer = TimestampSigner(b"secret-key", algorithm=algorithm)
    assert signer.algorithm == algorithm

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_with_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #34
#--------------------------

```python
def test_timestamp_signer_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #35
#--------------------------

```python
def test_timestamp_to_datetime_raises_oserror():
    signer = TimestampSigner("secret")
    with patch.object(signer, "timestamp_to_datetime", side_effect=OSError):
        with pytest.raises(BadTimeSignature) as exc_info:
            signer.unsign(b"value.sep.invalid_timestamp", return_timestamp=True)
    assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #36
#--------------------------

```python
def test_timestamp_to_datetime_raises_oserror():
    signer = TimestampSigner("secret")
    with patch.object(signer, "timestamp_to_datetime", side_effect=OSError("test error")):
        with raises(BadTimeSignature) as exc_info:
            signer.unsign("invalid.sig", return_timestamp=True)
        assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #37
#--------------------------

```python
def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    original_time = signer.get_timestamp()
    signed_value = signer.sign("test_value")
    signer.get_timestamp = lambda: original_time - 10
    try:
        signer.unsign(signed_value, max_age=5)
    except SignatureExpired as e:
        assert str(e) == f"Signature age -10 > 5 seconds"
        assert e.payload == b"test_value"
        assert e.date_signed == signer.timestamp_to_datetime(original_time)
    else:
        assert False, "Expected SignatureExpired exception"


# LLM-generated content at query #38
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data)
    assert result == "test_data"

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    try:
        serializer.loads("invalid_signature")
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    try:
        serializer.loads(signed_data, max_age=0)
        assert False, "Expected SignatureExpired exception"
    except SignatureExpired:
        pass

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == "test_data"
    assert isinstance(timestamp, int)

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data.encode("utf-8"))
    assert result == "test_data"

def test_loads_with_custom_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data", salt="custom_salt")
    result = serializer.loads(signed_data, salt="custom_salt")
    assert result == "test_data"


# LLM-generated content at query #39
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.algorithm.digest_method == Signer.default_digest_method

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom_secret",
        salt=b"custom_salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
    )
    assert signer.secret_keys == [b"custom_secret"]
    assert signer.salt == b"custom_salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.algorithm.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha512)
    signer = TimestampSigner(b"secret", algorithm=algorithm)
    assert signer.algorithm == algorithm
    assert signer.algorithm.digest_method == hashlib.sha512

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner([b"old_key", b"new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret", sep="a")


# LLM-generated content at query #40
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.invalid")
        assert False, "Expected BadSignature"
    except BadSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.malformed")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass


# LLM-generated content at query #41
#--------------------------

```python
def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret", salt="test")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == data
    assert isinstance(result[1], int)


# LLM-generated content at query #42
#--------------------------

```python
def test_unsign_without_separator_and_no_signature_error():
    signer = TimestampSigner("secret")
    result = b"value_without_separator"
    assert signer.sep not in result
    assert not hasattr(signer, "sig_error") or signer.sig_error is None


# LLM-generated content at query #43
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    signed_value = b"test.invalid_timestamp.invalid_signature"
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert e.payload == b"test.invalid_timestamp"

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"test.invalid_signature"
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert e.payload == b"test"

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    modified_signed_value = signed_value.replace(b".", b"X", 1)
    try:
        signer.unsign(modified_signed_value)
    except BadTimeSignature as e:
        assert e.payload == b"test"

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=-1)
    except SignatureExpired as e:
        assert e.payload == b"test"

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=0)
    except SignatureExpired as e:
        assert e.payload == b"test"


# LLM-generated content at query #44
#--------------------------

```python
def test_timestamp_to_datetime_raises_oserror():
    signer = TimestampSigner("secret")
    with patch.object(signer, "timestamp_to_datetime", side_effect=OSError):
        with pytest.raises(BadTimeSignature) as exc_info:
            signer.unsign("invalid")
        assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #45
#--------------------------

```python
def test_unsign_without_separator_and_no_signature_error():
    signer = TimestampSigner("secret")
    signed_value = b"value_without_separator"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert e.payload == signed_value
        assert str(e) == "timestamp missing"


# LLM-generated content at query #46
#--------------------------

```python
def test_unsign_raises_bad_time_signature_on_invalid_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it invalid
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    with pytest.raises(BadTimeSignature):
        signer.unsign(corrupted_value)


# LLM-generated content at query #47
#--------------------------

```python
def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2


# LLM-generated content at query #48
#--------------------------

```python
def test_loads_raises_signature_expired_immediately():
    serializer = TimedSerializer("secret")
    with pytest.raises(SignatureExpired):
        serializer.loads("expired_signature", max_age=0)


# LLM-generated content at query #49
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with patch.object(signer, 'get_timestamp', return_value=signer.get_timestamp() + 100):
        try:
            signer.unsign(signed, max_age=1)
            assert False, "Expected SignatureExpired"
        except SignatureExpired:
            pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with patch.object(signer, 'get_timestamp', return_value=signer.get_timestamp() - 100):
        try:
            signer.unsign(signed, max_age=1)
            assert False, "Expected SignatureExpired"
        except SignatureExpired:
            pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.sep.invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass


# LLM-generated content at query #50
#--------------------------

```python
def test_unsign_with_malformed_timestamp_does_not_set_ts_int():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature:
        pass
    # The test is to ensure that ts_int remains None when the timestamp is malformed
    # This is implicitly tested by the exception being raised


# LLM-generated content at query #51
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value

def test_unsign_with_valid_signature_and_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("test")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("test:invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)


# LLM-generated content at query #52
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.secret_keys == [b"custom-secret"]

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #53
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_max_age_exceeded():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test:invalid_timestamp")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)


# LLM-generated content at query #54
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"hello")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"hello.sep.invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)


# LLM-generated content at query #55
#--------------------------

```python
def test_unsign_without_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Manually remove the timestamp part to simulate missing timestamp
    modified_value = signed_value.rsplit(signer.sep.encode(), 1)[0]
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(modified_value)
    assert "timestamp missing" in str(exc_info.value)


# LLM-generated content at query #56
#--------------------------

```python
def test_timestamp_signer_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #57
#--------------------------

```python
def test_unsign_raises_signature_expired_for_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with patch.object(signer, "get_timestamp", return_value=0):
        with pytest.raises(SignatureExpired) as exc_info:
            signer.unsign(signed_value, max_age=10)
        assert "Signature age -" in str(exc_info.value)


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data)
    assert result == "test_data"

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    try:
        serializer.loads("invalid_data")
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    try:
        serializer.loads(signed_data, max_age=-1)
        assert False, "Expected SignatureExpired exception"
    except SignatureExpired:
        pass

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == "test_data"
    assert isinstance(timestamp, int)

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data.encode("utf-8"))
    assert result == "test_data"

def test_loads_with_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data", salt="custom_salt")
    result = serializer.loads(signed_data, salt="custom_salt")
    assert result == "test_data"


# LLM-generated content at query #2
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret")
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.secret_keys == [b"secret"]

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
    )
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.secret_keys == [b"custom-secret"]

def test_timestamp_signer_constructor_with_string_secret():
    signer = TimestampSigner("secret-string")
    assert signer.secret_keys == [b"secret-string"]

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha512)
    signer = TimestampSigner(b"secret", algorithm=algorithm)
    assert signer.algorithm is algorithm


# LLM-generated content at query #3
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    result = signer.unsign(signed_value)
    assert result == b"value"

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"value"
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value.sep.malformed_timestamp")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value_without_timestamp")

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value.sep.timestamp.invalid_signature")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)


# LLM-generated content at query #4
#--------------------------

```python
def test_unsign_with_missing_timestamp_and_no_signature_error():
    signer = TimestampSigner("secret")
    signed_value = b"value"
    assert not signer.unsign(signed_value)


# LLM-generated content at query #5
#--------------------------

```python
def test_unsign_with_invalid_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make base64_decode raise an exception
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert e.args[0] == "Malformed timestamp"
    else:
        assert False, "Expected BadTimeSignature to be raised"


# LLM-generated content at query #6
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Manually corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


# LLM-generated content at query #7
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_max_age_not_expired():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=1000) == value

def test_unsign_with_valid_signature_and_max_age_expired():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value" + signer.sep.encode())

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value" + signer.sep.encode() + b"invalid")

def test_unsign_with_return_timestamp_true():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

def test_unsign_with_return_timestamp_false():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result = signer.unsign(signed, return_timestamp=False)
    assert result == value
    assert not isinstance(result, tuple)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #8
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"malformed"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"


# LLM-generated content at query #9
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value"


# LLM-generated content at query #10
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_max_age_not_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=100) == value

def test_unsign_with_valid_signature_and_max_age_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.sep.invalid_timestamp")

def test_unsign_with_return_timestamp_true():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #11
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid")
        assert False, "Expected BadSignature"
    except BadSignature:
        pass

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.invalid_timestamp")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_unsign_valid_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_max_age_not_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=3600) == value

def test_unsign_with_max_age_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    pytest.raises(SignatureExpired, signer.unsign, signed, max_age=0)

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    pytest.raises(BadSignature, signer.unsign, b"invalid")

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    pytest.raises(BadTimeSignature, signer.unsign, b"test")

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign(b"test")
    # Corrupt the timestamp part
    corrupted = signed.rsplit(signer.sep.encode(), 1)[0] + b"invalid"
    pytest.raises(BadTimeSignature, signer.unsign, corrupted)


# LLM-generated content at query #13
#--------------------------

```python
def test_unsign_valid_signature_without_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    assert signer.unsign(signed_value) == b"value"

def test_unsign_valid_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    value, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert value == b"value"
    assert isinstance(timestamp, datetime)

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign("invalid_signature")

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign("value.sep.malformed_timestamp")

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign("value")

def test_unsign_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)


# LLM-generated content at query #14
#--------------------------

```python
def test_loads_raises_signature_expired_immediately():
    serializer = TimedSerializer("secret")
    with pytest.raises(SignatureExpired):
        serializer.loads("expired_signature", max_age=0)


# LLM-generated content at query #15
#--------------------------

```python
def test_unsign_valid_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_max_age_valid():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=3600) == value

def test_unsign_with_max_age_expired():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired as e:
        assert e.payload == value
        assert e.date_signed is not None

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
    except BadTimeSignature as e:
        assert e.payload == b"invalid"

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value.invalid_timestamp")
    except BadTimeSignature as e:
        assert e.payload == b"value"

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value_without_timestamp")
    except BadTimeSignature as e:
        assert e.payload == b"value_without_timestamp"

def test_unsign_negative_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired as e:
        assert e.payload == value
        assert e.date_signed is not None


# LLM-generated content at query #16
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
    )
    assert signer.secret_keys == [b"custom-secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner(b"secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_key_list():
    signer = TimestampSigner([b"key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #17
#--------------------------

```python
def test_unsign_with_invalid_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it invalid
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"


# LLM-generated content at query #18
#--------------------------

```python
def test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    payload, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(payload, str)
    assert isinstance(timestamp, int)


# LLM-generated content at query #19
#--------------------------

```python
def test_TimestampSigner_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_TimestampSigner_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=_lazy_sha256,
    )
    assert signer.secret_keys == [b"custom-secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == _lazy_sha256()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_TimestampSigner_constructor_with_list_of_keys():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_TimestampSigner_constructor_with_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used because it may be contained in the signature itself"):
        TimestampSigner(b"secret-key", sep=b"a")


# LLM-generated content at query #20
#--------------------------

```python
def test_loads_with_signature_expired_raises_immediately():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=-1)


# LLM-generated content at query #21
#--------------------------

```python
def test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == "data"
    assert isinstance(result[1], int)


# LLM-generated content at query #22
#--------------------------

```python
def test_loads_with_valid_signature_and_no_max_age():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result = serializer.loads(signed_data)
    assert result == "test data"

def test_loads_with_valid_signature_and_max_age_not_exceeded():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result = serializer.loads(signed_data, max_age=60)
    assert result == "test data"

def test_loads_with_valid_signature_and_return_timestamp():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == "test data"
    assert isinstance(timestamp, int)

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    with pytest.raises(BadSignature):
        serializer.loads("invalid signed data")

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result = serializer.loads(signed_data.encode("utf-8"))
    assert result == "test data"

def test_loads_with_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data", salt="custom-salt")
    result = serializer.loads(signed_data, salt="custom-salt")
    assert result == "test data"


# LLM-generated content at query #23
#--------------------------

```python
def test_loads_with_return_timestamp_true():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2


# LLM-generated content at query #24
#--------------------------

```python
def test_unsign_with_missing_timestamp_and_no_signature_error():
    signer = TimestampSigner("secret")
    result = b"value_without_timestamp"
    assert signer.sep not in result


# LLM-generated content at query #25
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
    )
    assert signer.secret_keys == [b"custom-secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner(b"secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_key_list():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_with_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #26
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
    )
    assert signer.secret_keys == [b"custom-secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner(b"secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret", sep="a")


# LLM-generated content at query #27
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid"
    try:
        signer.unsign(corrupted)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value"


# LLM-generated content at query #28
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    assert signer.unsign(signed_value) == b"value"

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    value, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert value == b"value"
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_signature"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.signature"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.malformed_timestamp.sep.signature"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    try:
        signer.unsign(signed_value, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    try:
        signer.unsign(signed_value, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass


# LLM-generated content at query #29
#--------------------------

```python
def test_unsign_without_sep_in_result_and_no_sig_error():
    signer = TimestampSigner("secret")
    result = b"value"
    sep = b"."

    assert sep not in result
    assert not hasattr(signer, "sig_error") or signer.sig_error is None

    try:
        signer.unsign(result)
    except BadTimeSignature as e:
        assert e.payload == result
        assert str(e) == "timestamp missing"


# LLM-generated content at query #30
#--------------------------

```python
def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2


# LLM-generated content at query #31
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test:malformed")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass


# LLM-generated content at query #32
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret")
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.secret_keys == [b"secret"]

def test_timestamp_signer_constructor_custom_params():
    signer = TimestampSigner(
        secret_key=b"custom_secret",
        salt=b"custom_salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.salt == b"custom_salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.secret_keys == [b"custom_secret"]

def test_timestamp_signer_constructor_with_string_secret():
    signer = TimestampSigner("secret_string")
    assert signer.secret_keys == [b"secret_string"]

def test_timestamp_signer_constructor_with_key_list():
    signer = TimestampSigner([b"key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret", sep="a")


# LLM-generated content at query #33
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.invalid_timestamp")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test")

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.wrong_signature")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #34
#--------------------------

```python
def test_unsign_without_separator_raises_bad_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadSignature):
        signer.unsign(b"value_without_separator")


# LLM-generated content at query #35
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
    )
    assert signer.secret_keys == [b"custom-secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="A")


# LLM-generated content at query #36
#--------------------------

```python
def test_unsign_raises_signature_expired_for_negative_age():
    signer = TimestampSigner("secret")
    timestamp = signer.get_timestamp() + 10
    signed_value = signer.sign(b"value")
    assert SignatureExpired is type(signer.unsign(signed_value, max_age=0))


# LLM-generated content at query #37
#--------------------------

```python
def test_timestamp_signer_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret-key", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_constructor_with_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_custom_digest_method():
    custom_digest = lambda x: x  # Mock digest method
    signer = TimestampSigner("secret-key", digest_method=custom_digest)
    assert signer.digest_method == custom_digest

def test_timestamp_signer_constructor_with_custom_algorithm():
    custom_algorithm = HMACAlgorithm(lambda x: x)  # Mock algorithm
    signer = TimestampSigner("secret-key", algorithm=custom_algorithm)
    assert signer.algorithm == custom_algorithm

def test_timestamp_signer_constructor_with_invalid_sep():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner("secret-key", sep="a")

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]


# LLM-generated content at query #38
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_max_age_not_expired():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=60) == value

def test_unsign_with_valid_signature_and_max_age_expired():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello.sep.invalid_timestamp")

def test_unsign_with_return_timestamp_true():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_return_timestamp_false():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result = signer.unsign(signed, return_timestamp=False)
    assert result == value
    assert not isinstance(result, tuple)

def test_unsign_with_negative_max_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #39
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_max_age_not_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=1000) == value

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.invalid_timestamp.invalid_sig")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.invalid_sig")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #40
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom_secret",
        salt=b"custom_salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=_lazy_sha256(),
        algorithm=HMACAlgorithm(_lazy_sha256())
    )
    assert signer.secret_keys == [b"custom_secret"]
    assert signer.salt == b"custom_salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == _lazy_sha256()
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #41
#--------------------------

```python
def test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data")
    payload, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(payload, str)
    assert isinstance(timestamp, int)


# LLM-generated content at query #42
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid")
        assert False, "Expected BadSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.sep.invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None


# LLM-generated content at query #43
#--------------------------

```python
def test_TimestampSigner_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #44
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_max_age_not_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=100) == value

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test" + b"." + b"malformed" + b"." + b"sig")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test" + b"." + b"sig")

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    future_timestamp = int(time.time()) + 100
    signed = value + b"." + base64_encode(int_to_bytes(future_timestamp)) + b"." + signer.get_signature(value + b"." + base64_encode(int_to_bytes(future_timestamp)))
    with pytest.raises(SignatureExpired):
        signer.unsign(signed)


# LLM-generated content at query #45
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data)
    assert result == "test_data"

def test_loads_with_valid_signature_and_max_age():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data, max_age=3600)
    assert result == "test_data"

def test_loads_with_valid_signature_and_return_timestamp():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == "test_data"
    assert isinstance(timestamp, int)

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data.encode("utf-8"))
    assert result == "test_data"

def test_loads_with_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data", salt="custom_salt")
    result = serializer.loads(signed_data, salt="custom_salt")
    assert result == "test_data"


# LLM-generated content at query #46
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


# LLM-generated content at query #47
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result = serializer.loads(signed_data)
    assert result == "test data"

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    try:
        serializer.loads("invalid signature")
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    try:
        serializer.loads(signed_data, max_age=-1)
        assert False, "Expected SignatureExpired exception"
    except SignatureExpired:
        pass

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == "test data"
    assert isinstance(timestamp, int)

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result = serializer.loads(signed_data.encode("utf-8"))
    assert result == "test data"

def test_loads_with_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data", salt="custom-salt")
    result = serializer.loads(signed_data, salt="custom-salt")
    assert result == "test data"


# LLM-generated content at query #48
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign(b"value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    assert noterraises(BadTimeSignature, signer.unsign, corrupted_value)


# LLM-generated content at query #49
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
    )
    assert signer.secret_keys == [b"custom-secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha512)
    signer = TimestampSigner(b"secret-key", algorithm=algorithm)
    assert signer.algorithm == algorithm

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #50
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    signed_value = signer.sign(value)
    result = signer.unsign(signed_value)
    assert result == value

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    signed_value = signer.sign(value)
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    signed_value = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    signed_value = value + b"." + b"invalid-timestamp" + b"." + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_value)

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    signed_value = value + b"." + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_value)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    signed_value = signer.sign(value)
    invalid_signed_value = signed_value[:-1] + b"x"
    with pytest.raises(BadTimeSignature):
        signer.unsign(invalid_signed_value)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    signed_value = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)


