####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
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
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #2
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result = serializer.loads(signed_data)
    assert result == "test data"

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    with pytest.raises(BadSignature):
        serializer.loads("invalid signature")

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=-1)

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

def test_loads_with_custom_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data", salt="custom-salt")
    result = serializer.loads(signed_data, salt="custom-salt")
    assert result == "test data"


# LLM-generated content at query #3
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    result = signer.unsign(signed_value)
    assert result == b"value"

def test_unsign_with_valid_signature_and_max_age_not_exceeded():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    result = signer.unsign(signed_value, max_age=100)
    assert result == b"value"

def test_unsign_with_valid_signature_and_max_age_exceeded():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

def test_unsign_with_valid_signature_and_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign("invalid_signature")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign("value")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign("value.sep.invalid_timestamp")

def test_unsign_with_return_timestamp_true():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"value"
    assert isinstance(timestamp, datetime)

def test_unsign_with_return_timestamp_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    result = signer.unsign(signed_value, return_timestamp=False)
    assert result == b"value"


# LLM-generated content at query #4
#--------------------------

```python
def test_loads_raises_signature_expired_immediately():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=-1)


# LLM-generated content at query #5
#--------------------------

```python
def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


# LLM-generated content at query #6
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
        assert "Signature age -" in str(e)

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

def test_unsign_with_valid_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=3600) == value


# LLM-generated content at query #7
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign(b"test")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


# LLM-generated content at query #8
#--------------------------

```python
def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    timestamp = signer.get_timestamp() + 10  # Future timestamp
    value = b"test_value"
    ts_bytes = base64_encode(int_to_bytes(timestamp))
    signed_value = value + b"." + ts_bytes + b"." + signer.get_signature(value + b"." + ts_bytes)
    try:
        signer.unsign(signed_value, max_age=5)
    except SignatureExpired as e:
        assert str(e) == f"Signature age {-10} < 0 seconds"


# LLM-generated content at query #9
#--------------------------

```python
def test_timestamp_to_datetime_raises_value_error():
    signer = TimestampSigner("secret")
    with pytest.raises(ValueError):
        signer.timestamp_to_datetime(999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999


# LLM-generated content at query #10
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value"


# LLM-generated content at query #11
#--------------------------

```python
def test_unsign_with_invalid_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make base64_decode fail
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert e.payload == b"value"
        assert "Malformed timestamp" in str(e)


# LLM-generated content at query #12
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

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

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"hello.invalid_timestamp")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass


# LLM-generated content at query #13
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
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")

def test_timestamp_signer_constructor_none_salt():
    signer = TimestampSigner(b"secret-key", salt=None)
    assert signer.salt == b"itsdangerous.TimestampSigner"


# LLM-generated content at query #14
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

def test_timestamp_signer_constructor_with_custom_parameters():
    signer = TimestampSigner(
        secret_key=["old-key", "new-key"],
        salt="custom-salt",
        sep="|",
        key_derivation="hmac",
        digest_method=_lazy_sha256,
        algorithm=HMACAlgorithm(_lazy_sha256)
    )
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == _lazy_sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #15
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_max_age_not_exceeded():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=3600) == value

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
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
        signer.unsign(b"value.invalid_timestamp")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value_without_timestamp")

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)


# LLM-generated content at query #16
#--------------------------

```python
def test_unsign_with_valid_signature():
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
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.sep.invalid")
    except BadTimeSignature:
        pass

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired:
        pass

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    result = signer.unsign(signed_value)
    assert result == b"value"

def test_unsign_with_valid_signature_and_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"value"
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    signed_value = signed_value[:-1] + b"x"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert e.payload == b"value"

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    signed_value = signed_value.split(signer.sep.encode())[0]
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    parts = signed_value.split(signer.sep.encode())
    signed_value = parts[0] + signer.sep.encode() + b"invalid"
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
    except SignatureExpired as e:
        assert e.payload == b"value"

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    try:
        signer.unsign(signed_value, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert e.payload == b"value"


# LLM-generated content at query #18
#--------------------------

```python
def test_unsign_with_valid_signature():
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
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

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


# LLM-generated content at query #19
#--------------------------

```python
def test_timestamp_to_datetime_raises_value_error():
    signer = TimestampSigner("secret")
    with patch.object(signer, "timestamp_to_datetime", side_effect=ValueError("Invalid timestamp")):
        with pytest.raises(BadTimeSignature) as exc_info:
            signer.unsign("value.sep.invalid_timestamp", return_timestamp=True)
        assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #20
#--------------------------

```python
def test_loads_raises_signature_expired():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=-1)


# LLM-generated content at query #21
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

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom_secret",
        salt=b"custom_salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.secret_keys == [b"custom_secret"]
    assert signer.salt == b"custom_salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old_key", b"new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret", sep="a")


# LLM-generated content at query #22
#--------------------------

```python
def test_unsign_without_separator_and_no_signature_error():
    signer = TimestampSigner("secret")
    result = b"value_without_separator"
    assert signer.sep not in result
    assert not hasattr(result, "payload")


# LLM-generated content at query #23
#--------------------------

```python
def test_unsign_with_invalid_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp.sep.signature"
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(signed_value)


# LLM-generated content at query #24
#--------------------------

```python
def test_timestamp_decode_failure():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp"
    assert signer.unsign(signed_value) == b"value"


# LLM-generated content at query #25
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    assert signer.unsign(signed) == b"value"

def test_unsign_with_valid_signature_and_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"value"
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid")
    except BadSignature:
        pass
    else:
        assert False, "Expected BadSignature"

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value")
    except BadTimeSignature:
        pass
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value.sep.invalid")
    except BadTimeSignature:
        pass
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired:
        pass
    else:
        assert False, "Expected SignatureExpired"

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired:
        pass
    else:
        assert False, "Expected SignatureExpired"


# LLM-generated content at query #26
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
        signer.unsign(b"invalid")
    except BadSignature:
        pass
    else:
        assert False, "Expected BadSignature"

def test_unsign_with_max_age_exceeded():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired:
        pass
    else:
        assert False, "Expected SignatureExpired"

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired:
        pass
    else:
        assert False, "Expected SignatureExpired"

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo == timezone.utc

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
    except BadTimeSignature:
        pass
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.sep.invalid")
    except BadTimeSignature:
        pass
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_string_input():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == b"test"


# LLM-generated content at query #27
#--------------------------

```python
def test_unsign_with_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(signed_value)
    assert exc_info.value.args[0] == "Malformed timestamp"


# LLM-generated content at query #28
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
    assert signer.get_timestamp() > 0

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
    assert signer.get_timestamp() > 0

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"


# LLM-generated content at query #29
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
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret-key", sep="|")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"|"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="concat")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_digest_method():
    signer = TimestampSigner("secret-key", digest_method=_lazy_sha256)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(_lazy_sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert signer.algorithm == algorithm

def test_timestamp_signer_constructor_with_key_list():
    signer = TimestampSigner(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #30
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_params():
    signer = TimestampSigner(
        secret_key=b"secret",
        salt=b"custom_salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"|"
    assert signer.salt == b"custom_salt"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old_secret", b"new_secret"])
    assert signer.secret_keys == [b"old_secret", b"new_secret"]
    assert signer.secret_key == b"new_secret"

def test_timestamp_signer_constructor_invalid_sep():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret", sep="a")

def test_timestamp_signer_constructor_none_salt():
    signer = TimestampSigner(b"secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #31
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

def test_timestamp_signer_constructor_custom_params():
    signer = TimestampSigner(
        secret_key=b"custom_secret",
        salt=b"custom_salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.secret_keys == [b"custom_secret"]
    assert signer.salt == b"custom_salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old_key", b"new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"


# LLM-generated content at query #32
#--------------------------

```python
def test_unsign_without_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    assert raises(BadTimeSignature, signer.unsign, "value")


# LLM-generated content at query #33
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_with_valid_signature_and_max_age():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadSignature):
        signer.unsign("invalid")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign("test")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign("test.sep.invalid_timestamp")

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)


# LLM-generated content at query #34
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"


# LLM-generated content at query #35
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Simulate a malformed timestamp by corrupting the timestamp part
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"


# LLM-generated content at query #36
#--------------------------

```python
def test_unsign_raises_signature_expired_for_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with patch.object(signer, "get_timestamp", return_value=0):
        with pytest.raises(SignatureExpired) as exc_info:
            signer.unsign(signed_value, max_age=10)
        assert "Signature age -1 < 0 seconds" in str(exc_info.value)


# LLM-generated content at query #37
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid")
        assert False, "Expected BadSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_max_age_exceeded():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"hello:invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass


# LLM-generated content at query #38
#--------------------------

```python
def test_timestamp_signer_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
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

def test_timestamp_signer_constructor_with_bytes_secret_key():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_constructor_with_list_of_keys():
    signer = TimestampSigner(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_timestamp_signer_constructor_with_invalid_separator():
    try:
        TimestampSigner("secret-key", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "The given separator cannot be used because it may be contained in the signature itself. ASCII letters, digits, and '-_=' must not be used."


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

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom_secret",
        salt=b"custom_salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.secret_keys == [b"custom_secret"]
    assert signer.salt == b"custom_salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old_key", b"new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret", sep="a")


# LLM-generated content at query #40
#--------------------------

```python
def test_timestamp_signer_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret-key", sep="|")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"|"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="concat")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_digest_method():
    custom_digest = lambda: hashlib.sha256
    signer = TimestampSigner("secret-key", digest_method=custom_digest)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == custom_digest()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_algorithm():
    custom_algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=custom_algorithm)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert signer.algorithm == custom_algorithm

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #41
#--------------------------

```python
def test_timestamp_decode_failure():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to cause base64_decode to fail
    corrupted = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid"
    try:
        signer.unsign(corrupted)
    except BadTimeSignature:
        pass
    assert True


# LLM-generated content at query #42
#--------------------------

```python
def test_timestamp_to_datetime_raises_oserror():
    signer = TimestampSigner("secret")
    with patch.object(signer, "timestamp_to_datetime", side_effect=OSError):
        with pytest.raises(BadTimeSignature) as exc_info:
            signer.unsign(b"value.sep.invalid_timestamp")
        assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #43
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
        signer.unsign(b"test.sep.invalid")
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
    class FutureSigner(TimestampSigner):
        def get_timestamp(self):
            return int(time.time()) + 100

    signer = FutureSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age -100 < 0 seconds" in str(e)

def test_unsign_with_valid_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=100) == value

def test_unsign_with_str_value():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == b"test"

def test_unsign_with_bytes_value():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value


# LLM-generated content at query #44
#--------------------------

```python
def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(signed_value)
    assert str(exc_info.value) == "Malformed timestamp"
    assert exc_info.value.payload == b"value"


# LLM-generated content at query #45
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
        b"secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_string_secret():
    signer = TimestampSigner("secret-string")
    assert signer.secret_keys == [b"secret-string"]

def test_timestamp_signer_constructor_with_key_list():
    signer = TimestampSigner([b"old-secret", b"new-secret"])
    assert signer.secret_keys == [b"old-secret", b"new-secret"]

def test_timestamp_signer_constructor_with_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret", sep="a")


# LLM-generated content at query #46
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid")
    except BadSignature:
        pass

def test_unsign_with_max_age_exceeded():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired:
        pass

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.sep.malformed")
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
    except BadTimeSignature:
        pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired:
        pass


# LLM-generated content at query #47
#--------------------------

```python
def test_timestamp_to_datetime_raises_oserror():
    signer = TimestampSigner("secret")
    with mock.patch.object(signer, "timestamp_to_datetime", side_effect=OSError):
        with pytest.raises(BadTimeSignature) as exc_info:
            signer.unsign(b"value.sep.invalid_timestamp", return_timestamp=True)
        assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #48
#--------------------------

```python
def test_timestamp_signer_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    current_time = signer.get_timestamp()
    future_time = current_time + 100
    signed_value = signer.sign_object({"data": "test", "time": future_time})
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed_value, max_age=0)
    assert exc_info.value.args[0] == f"Signature age {-100} < 0 seconds"


# LLM-generated content at query #49
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

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(_lazy_sha256)
    signer = TimestampSigner(b"secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #50
#--------------------------

```python
def test_timestamp_signer_constructor_with_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom-secret",
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=_lazy_sha256,
        algorithm=HMACAlgorithm(_lazy_sha256)
    )
    assert signer.secret_keys == [b"custom-secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == _lazy_sha256()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_with_invalid_separator():
    try:
        TimestampSigner(b"secret-key", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == "The given separator cannot be used because it may be contained in the signature itself. ASCII letters, digits, and '-_=' must not be used."


# LLM-generated content at query #51
#--------------------------

```python
def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b".invalid"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value"


# LLM-generated content at query #52
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    assert signer.unsign(signed) == b"value"

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    assert signer.unsign("invalid") == b""

def test_unsign_with_max_age_exceeded():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"value"
    assert isinstance(timestamp, datetime)

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value.invalid_timestamp")


# LLM-generated content at query #53
#--------------------------

```python
def test_timestamp_signer_constructor():
    signer = TimestampSigner("secret-key")
    assert isinstance(signer, TimestampSigner)
    assert isinstance(signer, Signer)
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #54
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
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
    with pytest.raises(BadSignature):
        signer.unsign("invalid_signature")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test" + b".")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test" + b"." + b"invalid_timestamp")

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)

def test_unsign_with_valid_signature_and_max_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, max_age=100)
    assert result == b"test"


# LLM-generated content at query #55
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


# LLM-generated content at query #56
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


# LLM-generated content at query #57
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")[:-1]  # Corrupt the signature to make it malformed
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_params():
    signer = TimestampSigner(
        secret_key=b"custom_secret",
        salt=b"custom_salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=_lazy_sha256,
    )
    assert signer.secret_keys == [b"custom_secret"]
    assert signer.sep == b"|"
    assert signer.salt == b"custom_salt"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == _lazy_sha256()
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(_lazy_sha256())
    signer = TimestampSigner(b"secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner([b"old_secret", b"new_secret"])
    assert signer.secret_keys == [b"old_secret", b"new_secret"]
    assert signer.secret_key == b"new_secret"

def test_timestamp_signer_constructor_with_invalid_separator():
    try:
        TimestampSigner(b"secret", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == (
            "The given separator cannot be used because it may be"
            " contained in the signature itself. ASCII letters,"
            " digits, and '-_=' must not be used."
        )


# LLM-generated content at query #2
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_max_age_not_exceeded():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=3600) == value

def test_unsign_with_valid_signature_and_max_age_exceeded():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired as e:
        assert e.payload == value

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
        signer.unsign(b"test" + signer.sep + b"invalid")
    except BadTimeSignature as e:
        assert e.payload == b"test"

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired as e:
        assert e.payload == value


# LLM-generated content at query #3
#--------------------------

```python
def test_unsign_without_separator_raises_bad_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadSignature):
        signer.unsign("invalid_signature_without_separator")


# LLM-generated content at query #4
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result = serializer.loads(signed_data)
    assert result == "test data"

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result = serializer.loads(signed_data.encode())
    assert result == "test data"

def test_loads_with_max_age_not_expired():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result = serializer.loads(signed_data, max_age=10)
    assert result == "test data"

def test_loads_with_max_age_expired():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    try:
        serializer.loads(signed_data, max_age=0)
        assert False, "Expected SignatureExpired exception"
    except SignatureExpired:
        pass

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == "test data"
    assert isinstance(timestamp, int)

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    try:
        serializer.loads("invalid signature")
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_loads_with_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data", salt="custom-salt")
    result = serializer.loads(signed_data, salt="custom-salt")
    assert result == "test data"


# LLM-generated content at query #5
#--------------------------

```python
def test_signature_expired_raises_immediately():
    serializer = TimedSerializer("secret")
    with pytest.raises(SignatureExpired):
        serializer.loads("expired_signature", max_age=0)


# LLM-generated content at query #6
#--------------------------

```python
def test_unsign_raises_signature_expired_for_negative_age():
    signer = TimestampSigner("secret")
    with patch.object(signer, "get_timestamp", return_value=100):
        signed_value = signer.sign("value")
    with patch.object(signer, "get_timestamp", return_value=50):
        with pytest.raises(SignatureExpired) as exc_info:
            signer.unsign(signed_value, max_age=100)
    assert exc_info.value.date_signed == signer.timestamp_to_datetime(100)


# LLM-generated content at query #7
#--------------------------

```python
def test_loads_with_valid_signature_and_no_max_age():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("data")
    assert serializer.loads(signed) == "data"

def test_loads_with_valid_signature_and_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("data")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "data"
    assert isinstance(timestamp, int)

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret")
    try:
        serializer.loads("invalid")
        assert False, "Expected BadSignature"
    except BadSignature:
        pass

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("data")
    try:
        serializer.loads(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("data")
    assert serializer.loads(signed.encode()) == "data"

def test_loads_with_custom_salt():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("data", salt="custom")
    assert serializer.loads(signed, salt="custom") == "data"


# LLM-generated content at query #8
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

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.sep.malformed")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test")

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.sep.12345.invalid")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #9
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
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
        signer.unsign(b"invalid")
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
        signer.unsign(b"hello.sep.invalid_timestamp")
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
    except SignatureExpired:
        pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_valid_signature_and_max_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=1000)
    assert result == value


# LLM-generated content at query #10
#--------------------------

```python
def test_loads_with_return_timestamp_true():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("test")
    result = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result, tuple) and len(result) == 2


# LLM-generated content at query #11
#--------------------------

```python
def test_timestamp_missing_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Manipulate the signed value to remove the timestamp
    sep = want_bytes(signer.sep)
    value, _ = signed_value.rsplit(sep, 1)
    manipulated_value = value + sep + signer.get_signature(value)

    try:
        signer.unsign(manipulated_value)
    except BadTimeSignature as e:
        assert e.payload == value
        assert str(e) == "timestamp missing"


# LLM-generated content at query #12
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    result = signer.unsign(signed_value)
    assert result == b"value"

def test_unsign_with_valid_signature_and_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"value"
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid_signature")
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    try:
        signer.unsign(signed_value, max_age=0)
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("value.sep.malformed_timestamp")
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    try:
        signer.unsign(signed_value, max_age=-1)
    except SignatureExpired as e:
        assert "Signature age" in str(e)


# LLM-generated content at query #13
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello.invalid_timestamp")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello")

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello.1234567890.invalid_signature")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #14
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
    signed_value = b"value" + b"." + b"invalid_timestamp" + b"." + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_value)

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"value" + b"." + b"signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_value)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    signed_value = b"value" + b"." + b"timestamp" + b"." + b"invalid_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_value)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)


# LLM-generated content at query #15
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_params():
    signer = TimestampSigner(
        secret_key=[b"old-key", b"new-key"],
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.sep == b"|"
    assert signer.salt == b"custom-salt"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #16
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret", salt="test_salt")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    result = serializer.loads(signed_data)
    assert result == data

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret", salt="test_salt")
    invalid_data = "invalid_data"
    try:
        serializer.loads(invalid_data)
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret", salt="test_salt")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    try:
        serializer.loads(signed_data, max_age=-1)
        assert False, "Expected SignatureExpired exception"
    except SignatureExpired:
        pass

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret", salt="test_salt")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, int)


# LLM-generated content at query #17
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

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
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

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.sep.invalid_timestamp.sep.signature")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.sep.signature")
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


# LLM-generated content at query #18
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b"invalid"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


# LLM-generated content at query #19
#--------------------------

```python
def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    assert raises(BadTimeSignature, signer.unsign, corrupted_value)


# LLM-generated content at query #20
#--------------------------

```python
def test_loads_with_valid_signature_and_no_max_age():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result = serializer.loads(signed_data)
    assert result == "data"

def test_loads_with_valid_signature_and_max_age_not_exceeded():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result = serializer.loads(signed_data, max_age=3600)
    assert result == "data"

def test_loads_with_valid_signature_and_return_timestamp():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == "data"
    assert isinstance(timestamp, int)

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret")
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result = serializer.loads(signed_data.encode("utf-8"))
    assert result == "data"

def test_loads_with_salt():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data", salt="salt")
    result = serializer.loads(signed_data, salt="salt")
    assert result == "data"

def test_loads_with_wrong_salt():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data", salt="salt")
    with pytest.raises(BadSignature):
        serializer.loads(signed_data, salt="wrong_salt")


# LLM-generated content at query #21
#--------------------------

```python
def test_unsign_raises_signature_expired_for_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    current_time = signer.get_timestamp()
    future_time = current_time + 100
    signer.get_timestamp = lambda: future_time
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed_value, max_age=50)
    assert "Signature age -50 < 0 seconds" in str(exc_info.value)


# LLM-generated content at query #22
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello.invalid_timestamp")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello")

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello.1234567890.invalid_signature")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #23
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
        secret_key=b"custom_secret",
        salt=b"custom_salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256),
    )
    assert signer.secret_keys == [b"custom_secret"]
    assert signer.salt == b"custom_salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old_secret", b"new_secret"])
    assert signer.secret_keys == [b"old_secret", b"new_secret"]
    assert signer.secret_key == b"new_secret"


# LLM-generated content at query #24
#--------------------------

```python
def test_signature_expired_exception_is_raised_immediately():
    serializer = TimedSerializer("secret")
    s = serializer.dumps("data")
    with pytest.raises(SignatureExpired):
        serializer.loads(s, max_age=-1)


# LLM-generated content at query #25
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value

def test_unsign_with_valid_signature_and_max_age_not_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=1000)
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

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test" + signer.sep + b"invalid_timestamp" + signer.sep + b"signature")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test" + signer.sep + b"signature")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #26
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data)
    assert result == "test_data"

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    with pytest.raises(BadSignature):
        serializer.loads("invalid_signature")

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=-1)

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


# LLM-generated content at query #27
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature exception"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"


# LLM-generated content at query #28
#--------------------------

```python
def test_timestamp_to_datetime_raises_oserror():
    signer = TimestampSigner("secret")
    with patch.object(signer, "timestamp_to_datetime", side_effect=OSError("Invalid timestamp")):
        with pytest.raises(BadTimeSignature) as exc_info:
            signer.unsign(b"value.sep.invalid_timestamp", return_timestamp=True)
        assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #29
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"


# LLM-generated content at query #30
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = "test_value"
    signed_value = signer.sign(value)
    result = signer.unsign(signed_value)
    assert result == b"test_value"

def test_unsign_with_valid_signature_and_max_age_not_exceeded():
    signer = TimestampSigner("secret")
    value = "test_value"
    signed_value = signer.sign(value)
    result = signer.unsign(signed_value, max_age=3600)
    assert result == b"test_value"

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = "test_value"
    signed_value = signer.sign(value)
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = "test_value"
    signed_value = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test_value:invalid_timestamp:signature")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test_value:signature")

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test_value:timestamp:invalid_signature")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = "test_value"
    signed_value = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)


# LLM-generated content at query #31
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
    signer = TimestampSigner("secret-key", digest_method=_lazy_sha256)
    assert signer.digest_method == _lazy_sha256

def test_timestamp_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(_lazy_sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm == algorithm

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"


# LLM-generated content at query #32
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


# LLM-generated content at query #33
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom_secret",
        salt=b"custom_salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=_lazy_sha256,
        algorithm=HMACAlgorithm(_lazy_sha256)
    )
    assert signer.secret_keys == [b"custom_secret"]
    assert signer.salt == b"custom_salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == _lazy_sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old_secret", b"new_secret"])
    assert signer.secret_keys == [b"old_secret", b"new_secret"]
    assert signer.secret_key == b"new_secret"

def test_timestamp_signer_constructor_invalid_separator():
    try:
        TimestampSigner(b"secret", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot be used" in str(e)


# LLM-generated content at query #34
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

def test_timestamp_signer_constructor_custom_params():
    signer = TimestampSigner(
        secret_key=[b"old-key", b"new-key"],
        salt=b"custom-salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256)
    )
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #35
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

def test_timestamp_signer_constructor_none_salt():
    signer = TimestampSigner(b"secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #36
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
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

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

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)


# LLM-generated content at query #37
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
    except BadSignature:
        pass
    else:
        assert False, "Expected BadSignature"

def test_unsign_with_max_age_exceeded():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired:
        pass
    else:
        assert False, "Expected SignatureExpired"

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
        signer.unsign(b"test:invalid_timestamp")
    except BadTimeSignature:
        pass
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
    except BadTimeSignature:
        pass
    else:
        assert False, "Expected BadTimeSignature"


# LLM-generated content at query #38
#--------------------------

```python
def test_loads_raises_signature_expired_immediately():
    serializer = TimedSerializer("secret")
    with pytest.raises(SignatureExpired):
        serializer.loads("expired_signature", max_age=0)


# LLM-generated content at query #39
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
        digest_method=_lazy_sha256,
    )
    assert signer.secret_keys == [b"custom-secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == _lazy_sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #40
#--------------------------

```python
def test_timestamp_signer_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1()
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
    custom_digest = lambda: hashlib.sha256
    signer = TimestampSigner("secret-key", digest_method=custom_digest())
    assert signer.digest_method == custom_digest()

def test_timestamp_signer_constructor_with_custom_algorithm():
    custom_algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=custom_algorithm)
    assert signer.algorithm == custom_algorithm

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_timestamp_signer_constructor_with_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner("secret-key", sep="a")


# LLM-generated content at query #41
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


# LLM-generated content at query #42
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

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.invalid_timestamp.signature")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.signature")

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.timestamp.invalid_signature")


# LLM-generated content at query #43
#--------------------------

```python
def test_unsign_raises_bad_time_signature_when_timestamp_is_malformed():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    assert raises(BadTimeSignature, signer.unsign, corrupted_value)


# LLM-generated content at query #44
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    result = serializer.loads(signed_data)
    assert result == "test data"

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data")
    from itsdangerous import SignatureExpired
    import pytest
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=0)

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    from itsdangerous import BadSignature
    import pytest
    with pytest.raises(BadSignature):
        serializer.loads("invalid signature")

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


# LLM-generated content at query #45
#--------------------------

```python
def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    future_timestamp = int(time.time()) + 100
    modified_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep + base64_encode(int_to_bytes(future_timestamp))
    try:
        signer.unsign(modified_value, max_age=50)
    except SignatureExpired as e:
        assert e.payload == b"value"
        assert str(e) == f"Signature age {-100} < 0 seconds"


# LLM-generated content at query #46
#--------------------------

```python
def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == "data"
    assert isinstance(result[1], int)


# LLM-generated content at query #47
#--------------------------

```python
def test_TimestampSigner_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_TimestampSigner_constructor_with_custom_sep():
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

def test_TimestampSigner_constructor_with_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_TimestampSigner_constructor_with_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_TimestampSigner_constructor_with_custom_digest_method():
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_TimestampSigner_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm == algorithm

def test_TimestampSigner_constructor_with_invalid_sep():
    with pytest.raises(ValueError):
        TimestampSigner("secret-key", sep="a")

def test_TimestampSigner_constructor_with_key_rotation():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"


# LLM-generated content at query #48
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_max_age_not_exceeded():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=3600) == value

def test_unsign_with_valid_signature_and_max_age_exceeded():
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
        signer.unsign(b"value")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value.sep.invalid_timestamp")

def test_unsign_with_return_timestamp_true():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #49
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign(b"value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value"


# LLM-generated content at query #50
#--------------------------

```python
def test_timestamp_to_datetime_raises_os_error():
    signer = TimestampSigner("secret")
    with patch.object(signer, "timestamp_to_datetime", side_effect=OSError):
        with pytest.raises(BadTimeSignature) as exc_info:
            signer.unsign("invalid.timestamp")
        assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #51
#--------------------------

```python
def test_timestamp_signer_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #52
#--------------------------

```python
def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    payload, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(payload, str)
    assert isinstance(timestamp, int)


# LLM-generated content at query #53
#--------------------------

```python
def test_timestamp_signer_constructor():
    secret_key = b"secret"
    salt = b"test_salt"
    sep = b"|"
    key_derivation = "hmac"
    digest_method = lambda x: __import__('hashlib').sha256(x)
    algorithm = HMACAlgorithm(digest_method)

    signer = TimestampSigner(
        secret_key=secret_key,
        salt=salt,
        sep=sep,
        key_derivation=key_derivation,
        digest_method=digest_method,
        algorithm=algorithm,
    )

    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"test_salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == digest_method
    assert signer.algorithm == algorithm


# LLM-generated content at query #54
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
        assert e.date_signed is not None
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
        signer.unsign(b"test.sep.invalid_timestamp")
    except BadTimeSignature as e:
        assert e.payload == b"test"
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_return_timestamp_true():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is timezone.utc

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired as e:
        assert e.payload == value
        assert e.date_signed is not None
    else:
        assert False, "Expected SignatureExpired"


# LLM-generated content at query #55
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp"
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert e.payload == b"value"
        assert str(e) == "Malformed timestamp"


# LLM-generated content at query #56
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
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test.invalid_timestamp")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #57
#--------------------------

```python
def test_returns_payload_and_timestamp_when_return_timestamp_is_true():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    result = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2


# LLM-generated content at query #58
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


# LLM-generated content at query #59
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
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

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello.sep.invalid_timestamp.sep.signature")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello.sep.signature")

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"hello.sep.timestamp.sep.invalid_signature")

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    future_timestamp = int(time.time()) + 1000
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)


# LLM-generated content at query #60
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


# LLM-generated content at query #61
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test.invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_max_age_exceeded():
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

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

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


# LLM-generated content at query #62
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

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
        signer.unsign(b"hello.sep.invalidts")
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
    except SignatureExpired:
        pass

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None


# LLM-generated content at query #63
#--------------------------

```python
def test_loads_with_signature_expired_raises_immediately():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=-1)


# LLM-generated content at query #64
#--------------------------

```python
def test_unsign_raises_bad_time_signature_for_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value"


# LLM-generated content at query #65
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

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign("invalid_signature")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign("value")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    malformed_value = signed_value[:-1] + b"X"
    with pytest.raises(BadTimeSignature):
        signer.unsign(malformed_value)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)


# LLM-generated content at query #66
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


# LLM-generated content at query #67
#--------------------------

```python
def test_timestamp_signer_constructor():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == _lazy_sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)


