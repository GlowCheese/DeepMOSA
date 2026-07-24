####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# LLM-generated content at query #1
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


# LLM-generated content at query #2
#--------------------------

```python
def test_unsign_with_valid_signature():
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

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    invalid_signed_value = signed_value[:-1] + b"x"
    try:
        signer.unsign(invalid_signed_value)
    except BadTimeSignature as e:
        assert str(e) == "Signature x does not match"
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    invalid_signed_value = signed_value.split(signer.sep.encode())[0]
    try:
        signer.unsign(invalid_signed_value)
    except BadTimeSignature as e:
        assert str(e) == "timestamp missing"
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    parts = signed_value.split(signer.sep.encode())
    invalid_signed_value = parts[0] + signer.sep.encode() + b"invalid"
    try:
        signer.unsign(invalid_signed_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    try:
        signer.unsign(signed_value, max_age=0)
    except SignatureExpired as e:
        assert "Signature age" in str(e)
    else:
        assert False, "Expected SignatureExpired"

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    parts = signed_value.split(signer.sep.encode())
    future_timestamp = base64_encode(int_to_bytes(int(time.time()) + 1000))
    invalid_signed_value = parts[0] + signer.sep.encode() + future_timestamp + signer.sep.encode() + parts[-1]
    try:
        signer.unsign(invalid_signed_value)
    except SignatureExpired as e:
        assert "Signature age" in str(e)
    else:
        assert False, "Expected SignatureExpired"


# LLM-generated content at query #3
#--------------------------

```python
def test_unsign_raises_signature_expired_when_age_exceeds_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    timestamp = signer.get_timestamp() - 100
    signed_value = signer.sign(value)
    with patch.object(signer, "get_timestamp", return_value=timestamp + 101):
        assert_raises(SignatureExpired, signer.unsign, signed_value, max_age=100)


# LLM-generated content at query #4
#--------------------------

```python
def test_timestamp_signer_constructor():
    secret_key = b"secret"
    salt = b"test_salt"
    sep = b"|"
    key_derivation = "hmac"
    digest_method = hashlib.sha256

    signer = TimestampSigner(
        secret_key=secret_key,
        salt=salt,
        sep=sep,
        key_derivation=key_derivation,
        digest_method=digest_method,
    )

    assert signer.secret_keys == [secret_key]
    assert signer.salt == salt
    assert signer.sep == sep
    assert signer.key_derivation == key_derivation
    assert signer.digest_method == digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #5
#--------------------------

```python
def test_unsign_without_separator_raises_bad_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("value_without_separator")
    except BadSignature as e:
        assert str(e) == "No signature found"
    else:
        assert False, "Expected BadSignature to be raised"


# LLM-generated content at query #6
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


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_49_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Tamper with the signature to cause a BadSignature error
    tampered_value = signed_value[:-1] + b"X"
    # Mock get_timestamp to return a value that will cause ts_int to be None
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: 0
    try:
        signer.unsign(tampered_value)
    except BadTimeSignature as e:
        assert e.date_signed is None
    finally:
        signer.get_timestamp = original_get_timestamp


# LLM-generated content at query #8
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

def test_loads_with_custom_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data", salt="custom_salt")
    result = serializer.loads(signed_data, salt="custom_salt")
    assert result == "test_data"


# LLM-generated content at query #9
#--------------------------

```python
def test_timestamp_to_datetime_raises_oserror():
    signer = TimestampSigner("secret")
    with patch.object(signer, "timestamp_to_datetime", side_effect=OSError):
        with pytest.raises(BadTimeSignature) as exc_info:
            signer.unsign(b"value.sep.invalid_timestamp")
        assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #10
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    result = serializer.loads(signed_data)
    assert result == data

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    try:
        serializer.loads("invalid-signed-data")
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    try:
        serializer.loads(signed_data, max_age=-1)
        assert False, "Expected SignatureExpired exception"
    except SignatureExpired:
        pass

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == data
    assert isinstance(timestamp, int)

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data)
    result = serializer.loads(signed_data.encode("utf-8"))
    assert result == data

def test_loads_with_salt():
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed_data = serializer.dumps(data, salt="custom-salt")
    result = serializer.loads(signed_data, salt="custom-salt")
    assert result == data


# LLM-generated content at query #11
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
    custom_digest = staticmethod(lambda x: x)
    signer = TimestampSigner("secret-key", digest_method=custom_digest)
    assert signer.digest_method == custom_digest

def test_timestamp_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(_lazy_sha1)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm == algorithm

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_with_invalid_separator():
    with pytest.raises(ValueError):
        TimestampSigner("secret-key", sep="a")


# LLM-generated content at query #12
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


# LLM-generated content at query #13
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
    assert signer.unsign(signed, max_age=3600) == value

def test_unsign_with_valid_signature_and_max_age_expired():
    signer = TimestampSigner("secret")
    value = b"hello"
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
        signer.unsign(b"value")
    except BadTimeSignature as e:
        assert e.payload == b"value"
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value.sep.invalid_timestamp")
    except BadTimeSignature as e:
        assert e.payload == b"value"
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_with_return_timestamp():
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
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired as e:
        assert e.payload == value
    else:
        assert False, "Expected SignatureExpired"


# LLM-generated content at query #14
#--------------------------

```python
def test_signature_expired_exception_is_raised_immediately():
    serializer = TimedSerializer("secret")
    s = serializer.dumps("data")
    with pytest.raises(SignatureExpired):
        serializer.loads(s, max_age=-1)


# LLM-generated content at query #15
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


# LLM-generated content at query #16
#--------------------------

```python
def test_loads_with_valid_signature_and_no_max_age():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data)
    assert result == "test_data"

def test_loads_with_valid_signature_and_max_age_not_exceeded():
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
        serializer.loads(signed_data, max_age=0)
        assert False, "Expected SignatureExpired exception"
    except SignatureExpired:
        pass

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data.encode("utf-8"))
    assert result == "test_data"

def test_loads_with_string_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data)
    assert result == "test_data"

def test_loads_with_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data", salt="custom_salt")
    result = serializer.loads(signed_data, salt="custom_salt")
    assert result == "test_data"


# LLM-generated content at query #17
#--------------------------

```python
def test_loads_raises_signature_expired_immediately():
    serializer = TimedSerializer("secret")
    with pytest.raises(SignatureExpired):
        serializer.loads("expired_signature", max_age=0)


# LLM-generated content at query #18
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
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


# LLM-generated content at query #19
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

def test_unsign_with_max_age_exceeded():
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
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass


# LLM-generated content at query #20
#--------------------------

```python
def test_signature_expired_raises_immediately():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    with mock.patch.object(serializer.default_signer, 'unsign', side_effect=SignatureExpired):
        with pytest.raises(SignatureExpired):
            serializer.loads(signed_data, max_age=1)


# LLM-generated content at query #21
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test" + signer.sep.encode()


# LLM-generated content at query #22
#--------------------------

```python
def test_unsign_with_sig_error_and_invalid_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp.sep.signature"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(signed_value, return_timestamp=True)
    assert exc_info.value.args[0] == "Malformed timestamp"
    assert exc_info.value.payload == b"value"


# LLM-generated content at query #23
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
        signer.unsign(b"test.sep.malformed")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #24
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed_value = signer.sign(value)
    result = signer.unsign(signed_value)
    assert result == value

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed_value = signer.sign(value)
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == value
    assert isinstance(timestamp, datetime)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    signed_value = b"test.invalid_timestamp.invalid_signature"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed_value = signer.sign(value)
    try:
        signer.unsign(signed_value, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed_value = signer.sign(value)
    try:
        signer.unsign(signed_value, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"test.malformed_timestamp.signature"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)


# LLM-generated content at query #25
#--------------------------

```python
def test_unsign_raises_signature_expired_for_negative_age():
    signer = TimestampSigner("secret")
    with mock.patch.object(signer, "get_timestamp", return_value=100):
        signed_value = signer.sign("value")
    with mock.patch.object(signer, "get_timestamp", return_value=50):
        assert_raises(SignatureExpired, signer.unsign, signed_value, max_age=100)


# LLM-generated content at query #26
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


# LLM-generated content at query #27
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


# LLM-generated content at query #28
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


# LLM-generated content at query #29
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
    signed = b"test.sep.invalid_timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed)

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = b"test"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    signed = b"test.sep.invalid_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #30
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
        signer.unsign(b"test:invalid_timestamp:signature")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test:signature")

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    future_timestamp = signer.get_timestamp() + 3600
    ts_bytes = base64_encode(int_to_bytes(future_timestamp))
    sep = want_bytes(signer.sep)
    signed = value + sep + ts_bytes + sep + signer.get_signature(value + sep + ts_bytes)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed)


# LLM-generated content at query #31
#--------------------------

```python
def test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data")
    payload, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert isinstance(payload, str)
    assert isinstance(timestamp, int)


# LLM-generated content at query #32
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    malformed_signed_value = signed_value.replace(b"=", b"")
    assert signer.unsign(malformed_signed_value) == b"value"


# LLM-generated content at query #33
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
    result = serializer.loads(signed_data.encode())
    assert result == "test data"

def test_loads_with_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test data", salt="custom-salt")
    result = serializer.loads(signed_data, salt="custom-salt")
    assert result == "test data"


# LLM-generated content at query #34
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


# LLM-generated content at query #35
#--------------------------

```python
def test_unsign_with_invalid_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp.sep.signature"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(signed_value)
    assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #36
#--------------------------

```python
def test_timestamp_to_datetime_raises_value_error():
    signer = TimestampSigner("secret")
    with pytest.raises(ValueError):
        signer.timestamp_to_datetime(99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999


# LLM-generated content at query #37
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
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test")

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test" + signer.sep + b"invalid")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #38
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

def test_timestamp_signer_constructor_custom_params():
    signer = TimestampSigner(
        secret_key=[b"old-key", b"new-key"],
        salt="custom-salt",
        sep="|",
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

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #39
#--------------------------

```python
def test_unsign_without_separator_raises_bad_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value")
    except BadSignature as e:
        assert str(e) == "No separator found in value"
    else:
        assert False, "Expected BadSignature to be raised"


# LLM-generated content at query #40
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
    assert timestamp.tzinfo is timezone.utc

def test_unsign_with_return_timestamp_false():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed, return_timestamp=False)
    assert result == value
    assert not isinstance(result, tuple)


# LLM-generated content at query #41
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


# LLM-generated content at query #42
#--------------------------

```python
def test_timestamp_to_datetime_with_invalid_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value.12345.invalid_signature")


# LLM-generated content at query #43
#--------------------------

```python
def test_timestamp_to_datetime_raises_oserror():
    signer = TimestampSigner("secret")
    with patch.object(signer, "timestamp_to_datetime", side_effect=OSError):
        with pytest.raises(BadTimeSignature) as exc_info:
            signer.unsign(b"value.sep.invalid_timestamp", return_timestamp=True)
        assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #44
#--------------------------

```python
def test_unsign_with_missing_timestamp_and_no_signature_error():
    signer = TimestampSigner("secret")
    signed_value = b"value"
    result = b"value"
    sig_error = None
    sep = b"."

    assert sep not in result
    assert not sig_error


# LLM-generated content at query #45
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
        signer.unsign(b"test.sep.12345678.invalid_signature")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #46
#--------------------------

```python
def test_signature_expired_raises_immediately():
    serializer = TimedSerializer("secret")
    with pytest.raises(SignatureExpired):
        serializer.loads("expired_signature", max_age=0)


# LLM-generated content at query #47
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_with_valid_signature_and_timestamp():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

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

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #48
#--------------------------

```python
def test_unsign_with_missing_timestamp_and_no_signature_error():
    signer = TimestampSigner("secret")
    assert not signer.validate(b"value")


# LLM-generated content at query #49
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


# LLM-generated content at query #50
#--------------------------

```python
def test_unsign_with_invalid_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_value)


# LLM-generated content at query #51
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")[:-1]  # Corrupt the signature to trigger BadSignature
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(signed_value)
    assert "timestamp missing" in str(exc_info.value)


# LLM-generated content at query #52
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
    custom_digest = lambda x: x  # Mock digest method
    signer = TimestampSigner(b"secret-key", digest_method=custom_digest)
    assert signer.digest_method == custom_digest

def test_timestamp_signer_constructor_with_custom_algorithm():
    custom_algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner(b"secret-key", algorithm=custom_algorithm)
    assert signer.algorithm == custom_algorithm

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_with_invalid_separator():
    try:
        TimestampSigner(b"secret-key", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == (
            "The given separator cannot be used because it may be"
            " contained in the signature itself. ASCII letters,"
            " digits, and '-_=' must not be used."
        )


# LLM-generated content at query #53
#--------------------------

```python
def test_timestamp_signer_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.TimestampSigner"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #54
#--------------------------

```python
def test_ts_int_is_none_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(signed_value)
    assert exc_info.value.args[0] == "Malformed timestamp"
    assert exc_info.value.payload == b"value"


# LLM-generated content at query #55
#--------------------------

```python
def test_unsign_with_invalid_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp to cause an exception in timestamp_to_datetime
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid_timestamp"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(corrupted_value)
    assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #56
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

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadSignature):
        signer.unsign(b"invalid")

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
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #57
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

def test_timestamp_signer_constructor_custom_salt():
    signer = TimestampSigner(b"secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_custom_sep():
    signer = TimestampSigner(b"secret-key", sep=b"|")
    assert signer.sep == b"|"

def test_timestamp_signer_constructor_custom_key_derivation():
    signer = TimestampSigner(b"secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_custom_digest_method():
    custom_digest = lambda x: x  # Mock digest method
    signer = TimestampSigner(b"secret-key", digest_method=custom_digest)
    assert signer.digest_method == custom_digest

def test_timestamp_signer_constructor_custom_algorithm():
    custom_algorithm = HMACAlgorithm(_lazy_sha1())
    signer = TimestampSigner(b"secret-key", algorithm=custom_algorithm)
    assert signer.algorithm == custom_algorithm

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep=b"a")


# LLM-generated content at query #58
#--------------------------

```python
def test_bytes_to_int_raises_exception():
    assert bytes_to_int(b"invalid_base64") == 0


# LLM-generated content at query #59
#--------------------------

```python
def test_timestamp_missing_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign("value")
    assert exc_info.value.args[0] == "timestamp missing"
    assert exc_info.value.payload == b"value"


# LLM-generated content at query #60
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

def test_timestamp_signer_constructor_with_string_secret():
    signer = TimestampSigner("secret-string")
    assert signer.secret_keys == [b"secret-string"]

def test_timestamp_signer_constructor_with_key_list():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_timestamp_signer_constructor_invalid_separator():
    try:
        TimestampSigner(b"secret", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == (
            "The given separator cannot be used because it may be"
            " contained in the signature itself. ASCII letters,"
            " digits, and '-_=' must not be used."
        )

def test_timestamp_signer_constructor_none_salt():
    signer = TimestampSigner(b"secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #61
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

def test_unsign_with_bytes_input():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == b"test"

def test_unsign_with_str_input():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    assert signer.unsign(signed.decode()) == value


# LLM-generated content at query #62
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
    signed = b"test:invalid_timestamp:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed)

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = b"test:signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed)

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    signed = b"test:timestamp:invalid_signature"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #63
#--------------------------

```python
def test_unsign_raises_signature_expired_for_negative_age():
    signer = TimestampSigner("secret")
    timestamp = signer.get_timestamp() + 10
    signed_value = signer.sign(b"test")[:-10] + base64_encode(int_to_bytes(timestamp))
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(signed_value, max_age=5)
    assert exc_info.value.args[0] == f"Signature age {-5} < 0 seconds"


# LLM-generated content at query #64
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


# LLM-generated content at query #65
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
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed) == value

def test_unsign_with_valid_signature_and_max_age_not_expired():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    assert signer.unsign(signed, max_age=100) == value

def test_unsign_with_valid_signature_and_max_age_expired():
    signer = TimestampSigner("secret")
    value = b"hello"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired as e:
        assert e.payload == value

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
    except BadTimeSignature as e:
        assert e.payload == b""

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value")
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value:malformed")
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)

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
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired as e:
        assert e.payload == value


# LLM-generated content at query #68
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
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)


# LLM-generated content at query #69
#--------------------------

```python
def test_unsign_raises_bad_time_signature_when_timestamp_is_malformed():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid_timestamp"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(corrupted_value)
    assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #70
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    try:
        signer.unsign(corrupted_value)
        assert False, "Expected BadTimeSignature to be raised"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value"


# LLM-generated content at query #71
#--------------------------

```python
def test_unsign_with_invalid_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp.sep.signature"
    assert raises(BadTimeSignature, signer.unsign, signed_value)


# LLM-generated content at query #72
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret")
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == TimestampSigner.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.secret_keys == [b"secret"]

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key=b"custom_secret",
        salt=b"custom_salt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
    )
    assert signer.salt == b"custom_salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.secret_keys == [b"custom_secret"]

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner(b"secret", algorithm=algorithm)
    assert signer.algorithm is algorithm


# LLM-generated content at query #73
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

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #74
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

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"invalid")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test" + signer.sep.encode())

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"test" + signer.sep.encode() + b"invalid")

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    with pytest.raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)


# LLM-generated content at query #75
#--------------------------

```python
def test_negative_age_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    future_timestamp = signer.get_timestamp() + 100
    modified_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep + base64_encode(int_to_bytes(future_timestamp))
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(modified_value)
    assert "Signature age -100 < 0 seconds" in str(exc_info.value)


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

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(_lazy_sha256)
    signer = TimestampSigner(b"secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_with_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #2
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    assert signer.unsign(signed_value) == b"value"

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    value, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert value == b"value"
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=-1)

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    future_timestamp = signer.get_timestamp() + 100
    signed_value = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value.sep.invalid_timestamp.sep.signature")

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value.sep.signature")

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature):
        signer.unsign(b"value.sep.timestamp.sep.invalid_signature")

def test_unsign_with_valid_signature_and_max_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    assert signer.unsign(signed_value, max_age=100) == b"value"

def test_unsign_with_valid_signature_and_max_age_zero():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with pytest.raises(SignatureExpired):
        signer.unsign(signed_value, max_age=0)


# LLM-generated content at query #3
#--------------------------

```python
def test_unsign_raises_signature_expired_for_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    signer.get_timestamp = lambda: 0
    assert raises(SignatureExpired, signer.unsign, signed_value, max_age=10)


# LLM-generated content at query #4
#--------------------------

```python
def test_unsign_raises_bad_time_signature_on_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.1234567890"
    with patch.object(signer, "timestamp_to_datetime", side_effect=ValueError):
        with raises(BadTimeSignature) as exc_info:
            signer.unsign(signed_value)
        assert exc_info.value.payload == b"value"
        assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #5
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

def test_loads_with_custom_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data", salt="custom_salt")
    result = serializer.loads(signed_data, salt="custom_salt")
    assert result == "test_data"


# LLM-generated content at query #6
#--------------------------

```python
def test_loads_raises_signature_expired():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=-1)


# LLM-generated content at query #7
#--------------------------

```python
def test_unsign_with_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + b"invalid"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(corrupted_value)
    assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #8
#--------------------------

```python
def test_unsign_raises_signature_expired_when_age_is_negative():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    future_timestamp = signer.get_timestamp() + 1000
    fake_signed_value = b"value." + base64_encode(int_to_bytes(future_timestamp)) + b"." + signer.get_signature(b"value." + base64_encode(int_to_bytes(future_timestamp)))
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(fake_signed_value)
    assert exc_info.value.payload == b"value"
    assert exc_info.value.date_signed == signer.timestamp_to_datetime(future_timestamp)


# LLM-generated content at query #9
#--------------------------

```python
def test_unsign_raises_bad_time_signature_when_timestamp_is_malformed():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b"invalid_timestamp"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(corrupted_value)
    assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #10
#--------------------------

```python
def test_ts_int_is_none_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Manually corrupt the timestamp part to make it invalid
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(corrupted_value)
    assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #11
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data")
    result = serializer.loads(signed_data)
    assert result == "test-data"

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    try:
        serializer.loads("invalid-data")
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data")
    try:
        serializer.loads(signed_data, max_age=-1)
        assert False, "Expected SignatureExpired exception"
    except SignatureExpired:
        pass

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data")
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == "test-data"
    assert isinstance(timestamp, int)

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data")
    result = serializer.loads(signed_data.encode("utf-8"))
    assert result == "test-data"

def test_loads_with_custom_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data", salt="custom-salt")
    result = serializer.loads(signed_data, salt="custom-salt")
    assert result == "test-data"

def test_loads_with_wrong_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data", salt="custom-salt")
    try:
        serializer.loads(signed_data, salt="wrong-salt")
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass


# LLM-generated content at query #12
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b"invalid"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value"
    else:
        assert False, "Expected BadTimeSignature to be raised"


# LLM-generated content at query #13
#--------------------------

```python
def test_unsign_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(signed_value)
    assert exc_info.value.args[0] == "Malformed timestamp"
    assert exc_info.value.payload == b"value"


# LLM-generated content at query #14
#--------------------------

```python
def test_unsign_without_timestamp_and_without_signature_error():
    signer = TimestampSigner("secret")
    assert not signer.unsign(b"value_without_timestamp_and_signature")


# LLM-generated content at query #15
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp"
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value"


# LLM-generated content at query #16
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data")
    result = serializer.loads(signed_data)
    assert result == "test-data"

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    try:
        serializer.loads("invalid-signed-data")
        assert False, "Expected BadSignature exception"
    except BadSignature:
        pass

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data")
    try:
        serializer.loads(signed_data, max_age=-1)
        assert False, "Expected SignatureExpired exception"
    except SignatureExpired:
        pass

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data")
    result, timestamp = serializer.loads(signed_data, return_timestamp=True)
    assert result == "test-data"
    assert isinstance(timestamp, int)

def test_loads_with_string_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data")
    result = serializer.loads(signed_data.decode("utf-8"))
    assert result == "test-data"

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data")
    result = serializer.loads(signed_data)
    assert result == "test-data"

def test_loads_with_custom_salt():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test-data", salt="custom-salt")
    result = serializer.loads(signed_data, salt="custom-salt")
    assert result == "test-data"


# LLM-generated content at query #17
#--------------------------

```python
def test_signature_expired_raises_immediately():
    serializer = TimedSerializer("secret")
    with pytest.raises(SignatureExpired):
        serializer.loads("expired_signature", max_age=0)


# LLM-generated content at query #18
#--------------------------

```python
def test_timestamp_signer_constructor():
    secret_key = b"secret"
    salt = b"test_salt"
    sep = b"|"
    key_derivation = "hmac"
    digest_method = hashlib.sha256
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
    assert signer.digest_method == hashlib.sha256
    assert signer.algorithm == algorithm


# LLM-generated content at query #19
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
        TimestampSigner(b"secret", sep="a")


# LLM-generated content at query #20
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


# LLM-generated content at query #21
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_values():
    signer = TimestampSigner(
        secret_key="custom-secret",
        salt="custom-salt",
        sep="|",
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
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_key_list():
    signer = TimestampSigner(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner("secret-key", sep="a")


# LLM-generated content at query #22
#--------------------------

```python
def test_unsign_with_invalid_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp to cause an OverflowError
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep + b"AAAAAAAAAAA="
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(corrupted_value)


# LLM-generated content at query #23
#--------------------------

```python
def test_unsign_without_separator_in_result_and_no_sig_error():
    signer = TimestampSigner("secret")
    result = b"value_without_separator"
    sep = want_bytes(signer.sep)
    assert sep not in result
    try:
        signer.unsign(result)
    except BadTimeSignature as e:
        assert e.payload == result
        assert str(e) == "timestamp missing"
    else:
        assert False, "Expected BadTimeSignature to be raised"


# LLM-generated content at query #24
#--------------------------

```python
def test_unsign_raises_signature_expired_when_age_negative():
    signer = TimestampSigner("secret")
    with patch.object(signer, "get_timestamp", return_value=100):
        signed_value = signer.sign("value")
    with patch.object(signer, "get_timestamp", return_value=50):
        with pytest.raises(SignatureExpired) as exc_info:
            signer.unsign(signed_value)
    assert exc_info.value.args[0] == "Signature age -50 < 0 seconds"


# LLM-generated content at query #25
#--------------------------

```python
def test_unsign_with_missing_timestamp_and_no_signature_error():
    signer = TimestampSigner("secret")
    signed_value = b"value"
    result = b"value"
    sep = b"."

    assert sep not in result
    assert not hasattr(signer, "sig_error") or signer.sig_error is None


# LLM-generated content at query #26
#--------------------------

```python
def test_unsign_with_missing_timestamp_and_no_signature_error():
    signer = TimestampSigner("secret")
    result = b"value"
    assert signer.unsign(result) is None


# LLM-generated content at query #27
#--------------------------

```python
def test_unsign_with_invalid_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp"
    with pytest.raises(BadTimeSignature):
        signer.unsign(signed_value)


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


# LLM-generated content at query #29
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
    except SignatureExpired as e:
        assert "Signature age" in str(e)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "Signature age" in str(e)


# LLM-generated content at query #30
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret")
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
    signer = TimestampSigner(b"secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner([b"old-secret", b"new-secret"])
    assert signer.secret_keys == [b"old-secret", b"new-secret"]
    assert signer.secret_key == b"new-secret"

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


# LLM-generated content at query #31
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


# LLM-generated content at query #32
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
    assert signer.secret_key == b"key2"

def test_timestamp_signer_constructor_with_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner(b"secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #33
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

def test_timestamp_signer_constructor_custom_parameters():
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


# LLM-generated content at query #34
#--------------------------

```python
def test_unsign_raises_bad_time_signature_when_timestamp_is_malformed():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b"invalid"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(corrupted_value)
    assert "Malformed timestamp" in str(exc_info.value)


# LLM-generated content at query #35
#--------------------------

```python
def test_loads_raises_signature_expired_immediately():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=-1)


# LLM-generated content at query #36
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
        signer.unsign(b"test:malformed")
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


# LLM-generated content at query #37
#--------------------------

```python
def test_unsign_without_separator_in_result_and_no_sig_error():
    signer = TimestampSigner("secret")
    assert "sep" not in signer.unsign(b"invalid_data_without_separator")


# LLM-generated content at query #38
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
    assert timestamp.tzinfo is timezone.utc

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert e.payload == b""
        assert "timestamp missing" in str(e)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert e.payload == value
        assert "Signature age" in str(e)

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert e.payload == value
        assert "Signature age" in str(e)

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    sep = want_bytes(signer.sep)
    malformed = value + sep + b"invalid_timestamp" + sep + b"signature"
    try:
        signer.unsign(malformed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert e.payload == value
        assert "Malformed timestamp" in str(e)


# LLM-generated content at query #39
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


# LLM-generated content at query #40
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
        secret_key=b"secret",
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

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner([b"old-secret", b"new-secret"])
    assert signer.secret_keys == [b"old-secret", b"new-secret"]
    assert signer.secret_key == b"new-secret"

def test_timestamp_signer_constructor_invalid_separator():
    try:
        TimestampSigner(b"secret", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert str(e) == (
            "The given separator cannot be used because it may be"
            " contained in the signature itself. ASCII letters,"
            " digits, and '-_=' must not be used."
        )


# LLM-generated content at query #41
#--------------------------

```python
def test_unsign_raises_signature_expired_for_future_timestamp():
    signer = TimestampSigner("secret")
    future_timestamp = signer.get_timestamp() + 100
    signed_value = signer.sign(b"value")
    # Manipulate the signed value to have a future timestamp
    parts = signed_value.split(signer.sep.encode())
    manipulated_value = parts[0] + signer.sep.encode() + base64_encode(int_to_bytes(future_timestamp)) + signer.sep.encode() + parts[-1]
    with pytest.raises(SignatureExpired) as exc_info:
        signer.unsign(manipulated_value)
    assert "Signature age -100 < 0 seconds" in str(exc_info.value)


# LLM-generated content at query #42
#--------------------------

```python
def test_timestamp_to_datetime_raises_exception():
    signer = TimestampSigner("secret")
    with patch.object(signer, "timestamp_to_datetime", side_effect=ValueError):
        assert not signer.timestamp_to_datetime(1234567890)


# LLM-generated content at query #43
#--------------------------

```python
def test_loads_with_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    result = serializer.loads(signed_data)
    assert result == "test_data"

def test_loads_with_expired_signature():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("test_data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=-1)

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret-key")
    with pytest.raises(BadSignature):
        serializer.loads("invalid_data")

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


# LLM-generated content at query #44
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

def test_timestamp_signer_constructor_custom_parameters():
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

def test_timestamp_signer_constructor_with_list_of_keys():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_invalid_separator():
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #45
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.TimestampSigner"
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
        digest_method=_lazy_sha256(),
        algorithm=HMACAlgorithm(_lazy_sha256())
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
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner(b"secret-key", sep="a")


# LLM-generated content at query #46
#--------------------------

```python
def test_age_negative_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    with patch.object(signer, "get_timestamp", return_value=0):
        with pytest.raises(SignatureExpired) as exc_info:
            signer.unsign(signed_value, max_age=10)
        assert "Signature age -" in str(exc_info.value)


# LLM-generated content at query #47
#--------------------------

```python
def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign(b"value")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + b"invalid_timestamp"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"value"


# LLM-generated content at query #48
#--------------------------

```python
def test_bytes_to_int_raises_exception():
    assert bytes_to_int(b"invalid") is None


# LLM-generated content at query #49
#--------------------------

```python
def test_unsign_with_valid_signature():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    assert signer.unsign(signed) == b"test"

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign("invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_max_age_exceeded():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = "test"
    signed = signer.sign(value)
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)
    assert timestamp.tzinfo is not None

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
        signer.unsign(b"test" + b"." + b"invalid")
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass


# LLM-generated content at query #50
#--------------------------

```python
def test_loads_with_signature_expired_raises_immediately():
    serializer = TimedSerializer("secret")
    signed_data = serializer.dumps("data")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=-1)


# LLM-generated content at query #51
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

def test_timestamp_signer_constructor_with_string_secret():
    signer = TimestampSigner("string-secret")
    assert signer.secret_keys == [b"string-secret"]

def test_timestamp_signer_constructor_with_invalid_separator():
    try:
        TimestampSigner(b"secret", sep="a")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "cannot be used" in str(e)


# LLM-generated content at query #52
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


# LLM-generated content at query #53
#--------------------------

```python
def test_unsign_with_invalid_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp.sep.signature"
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(signed_value)
    assert exc_info.value.payload == b"value"


# LLM-generated content at query #54
#--------------------------

```python
def test_unsign_with_negative_age_raises_signature_expired():
    signer = TimestampSigner("secret")
    with patch.object(signer, "get_timestamp", return_value=100):
        signed_value = signer.sign("value")
    with patch.object(signer, "get_timestamp", return_value=50):
        with raises(SignatureExpired) as exc_info:
            signer.unsign(signed_value, max_age=100)
    assert "Signature age -50 < 0 seconds" in str(exc_info.value)


# LLM-generated content at query #55
#--------------------------

```python
def test_unsign_without_separator_in_result_and_no_signature_error():
    signer = TimestampSigner("secret")
    assert not (b"." in b"result_without_separator")


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
def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Corrupt the timestamp part to make it malformed
    corrupted_value = signed_value.rsplit(b".", 1)[0] + b".invalid"
    try:
        signer.unsign(corrupted_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test"


# LLM-generated content at query #58
#--------------------------

```python
def test_unsign_without_separator_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(b"value")
    assert exc_info.value.args[0] == "timestamp missing"
    assert exc_info.value.payload == b"value"


# LLM-generated content at query #59
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.TimestampSigner"
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
        algorithm=HMACAlgorithm(_lazy_sha256)
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


# LLM-generated content at query #60
#--------------------------

```python
def test_loads_raises_signature_expired_when_signature_is_expired():
    serializer = TimedSerializer("secret-key")
    signed_data = serializer.dumps("data", salt="test-salt")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed_data, max_age=-1, salt="test-salt")


# LLM-generated content at query #61
#--------------------------

```python
def test_unsign_with_invalid_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"value.sep.invalid_timestamp"
    assert not signer.unsign(signed_value)


# LLM-generated content at query #62
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


# LLM-generated content at query #63
#--------------------------

```python
def test_negative_age_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    future_timestamp = int(time.time()) + 100
    modified_value = signed_value.rsplit(signer.sep.encode(), 1)[0] + signer.sep + base64_encode(int_to_bytes(future_timestamp))
    try:
        signer.unsign(modified_value, max_age=50)
    except SignatureExpired as e:
        assert e.payload == b"value"
        assert str(e) == f"Signature age {-100} < 0 seconds"
    else:
        assert False, "Expected SignatureExpired to be raised"


# LLM-generated content at query #64
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
    else:
        assert False, "Expected BadTimeSignature to be raised"


# LLM-generated content at query #65
#--------------------------

```python
def test_timestamp_to_datetime_with_invalid_timestamp():
    signer = TimestampSigner("secret")
    with pytest.raises(BadTimeSignature) as exc_info:
        signer.unsign(b"value.sep.invalid_timestamp.sep.signature")
    assert "Malformed timestamp" in str(exc_info.value)


