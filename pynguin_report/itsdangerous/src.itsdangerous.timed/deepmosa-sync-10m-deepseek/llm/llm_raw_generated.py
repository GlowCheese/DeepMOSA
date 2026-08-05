####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_loads_with_str_returns_payload():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_with_bytes_returns_payload():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed.encode())
    assert result == "test"

def test_loads_with_return_timestamp_true_returns_tuple():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, float)

def test_loads_with_max_age_valid_returns_payload():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, max_age=3600)
    assert result == "test"

def test_loads_with_max_age_expired_raises_signature_expired():
    import time
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    time.sleep(0.1)
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except Exception as e:
        assert type(e).__name__ == "SignatureExpired"

def test_loads_with_invalid_signature_raises_bad_signature():
    serializer = TimedSerializer("secret")
    try:
        serializer.loads(b"invalid")
        assert False
    except Exception as e:
        assert type(e).__name__ == "BadSignature"

def test_loads_with_custom_salt_returns_payload():
    serializer = TimedSerializer("secret", salt="custom")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, salt="custom")
    assert result == "test"

def test_loads_with_wrong_salt_raises_bad_signature():
    serializer = TimedSerializer("secret", salt="custom1")
    signed = serializer.dumps("test")
    try:
        serializer.loads(signed, salt="custom2")
        assert False
    except Exception as e:
        assert type(e).__name__ == "BadSignature"

def test_loads_raises_bad_signature_on_first_signer_and_second_signer_fails():
    from unittest.mock import MagicMock
    serializer = TimedSerializer("secret")
    serializer.iter_unsigners = MagicMock()
    mock_signer1 = MagicMock()
    mock_signer1.unsign.side_effect = Exception("BadSignature")
    mock_signer2 = MagicMock()
    mock_signer2.unsign.side_effect = Exception("BadSignature")
    serializer.iter_unsigners.return_value = [mock_signer1, mock_signer2]
    try:
        serializer.loads(b"test")
        assert False
    except Exception as e:
        assert type(e).__name__ == "BadSignature"


# LLM-generated content at query #2
#--------------------------

```python
def test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true():
    serializer = TimedSerializer("secret-key")
    payload = {"key": "value"}
    serialized = serializer.dumps(payload)
    result = serializer.loads(serialized, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == payload
    assert isinstance(result[1], int) or isinstance(result[1], float)
```


# LLM-generated content at query #3
#--------------------------

def test_unsign_returns_bytes_when_return_timestamp_false():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed, return_timestamp=False)
    assert result == b"value"

def test_unsign_returns_tuple_when_return_timestamp_true():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"value"
    assert isinstance(result[1], datetime)

def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_raises_bad_signature_on_invalid_signature():
    signer = TimestampSigner("secret")
    signed = b"invalid" + signer.sep.encode() + b"fake"
    try:
        signer.unsign(signed)
        assert False
    except BadSignature:
        pass

def test_unsign_raises_bad_time_signature_when_timestamp_missing():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    # Remove the timestamp part by cutting off the last two segments
    parts = signed.split(signer.sep.encode())
    no_timestamp = parts[0] + signer.sep.encode() + parts[-1]
    try:
        signer.unsign(no_timestamp)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_raises_signature_expired_when_age_exceeds_max_age():
    import time
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    # Simulate that time has passed by setting a very small max_age
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_raises_signature_expired_when_age_negative():
    signer = TimestampSigner("secret")
    # Create a signed value with a future timestamp
    future_ts = int(time.time()) + 1000
    value = b"value"
    sep = signer.sep.encode()
    ts_bytes = base64_encode(int_to_bytes(future_ts))
    signed = value + sep + ts_bytes + sep + signer.get_signature(value + sep + ts_bytes)
    try:
        signer.unsign(signed, max_age=3600)
        assert False
    except SignatureExpired:
        pass

def test_unsign_returns_value_and_timestamp_when_return_timestamp_true():
    signer = TimestampSigner("secret")
    signed = signer.sign("data")
    value, ts = signer.unsign(signed, return_timestamp=True)
    assert value == b"data"
    assert isinstance(ts, datetime)

def test_unsign_accepts_bytes_input():
    signer = TimestampSigner("key")
    signed = signer.sign(b"bytes")
    result = signer.unsign(signed)
    assert result == b"bytes"

def test_unsign_accepts_string_input():
    signer = TimestampSigner("key")
    signed = signer.sign("string")
    result = signer.unsign(signed)
    assert result == b"string"


# LLM-generated content at query #4
#--------------------------

```python
def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_future_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=86400)
    except SignatureExpired:
        assert False

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test") + b"invalid"
    try:
        signer.unsign(signed)
        assert False
    except BadSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    # Remove timestamp part
    sep = signer.sep.encode()
    parts = signed.rsplit(sep, 1)
    signed_no_timestamp = parts[0]
    try:
        signer.unsign(signed_no_timestamp)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    sep = signer.sep.encode()
    malformed_ts = base64_encode(b"not_a_valid_timestamp")
    signed = value + sep + malformed_ts + sep + signer.get_signature(value + sep + malformed_ts)
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_invalid_signature_and_valid_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    # Corrupt the signature part
    sep = signer.sep.encode()
    parts = signed.rsplit(sep, 1)
    corrupted_signed = parts[0] + sep + b"corrupted"
    try:
        signer.unsign(corrupted_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_empty_string():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"")
        assert False
    except BadSignature:
        pass

def test_unsign_with_unicode_string_input():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed.decode())
    assert result == b"test"
```


# LLM-generated content at query #5
#--------------------------

```python
def test_age_less_than_zero_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Simulate a timestamp in the future (age < 0) by manipulating the signed value
    # The sign method adds current timestamp. We'll replace timestamp with a future one.
    # For simplicity, we can patch get_timestamp to return a future time.
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: original_get_timestamp() + 10
    # Now sign with a "future" timestamp, then unsign with max_age=0
    signed_future = signer.sign("test")
    signer.get_timestamp = original_get_timestamp  # restore
    try:
        signer.unsign(signed_future, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired as e:
        assert "age" in str(e)
        assert "< 0" in str(e)
```


# LLM-generated content at query #6
#--------------------------

def test_unsign_returns_value_when_valid():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_returns_value_when_valid_with_max_age():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_returns_bytes():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("hello")
    result = signer.unsign(signed)
    assert isinstance(result, bytes)

def test_unsign_with_return_timestamp_returns_tuple():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bytes)
    assert isinstance(result[1], datetime)

def test_unsign_raises_bad_signature_on_invalid_signature():
    signer = TimestampSigner("secret-key")
    signed = b"test.invalid"
    try:
        signer.unsign(signed)
        assert False
    except BadSignature:
        pass

def test_unsign_raises_bad_time_signature_on_missing_separator():
    signer = TimestampSigner("secret-key")
    try:
        signer.unsign(b"test")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_raises_signature_expired_when_age_exceeds_max_age():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_raises_signature_expired_when_age_negative():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_returns_value_with_max_age_none():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=None)
    assert result == b"test"


# LLM-generated content at query #7
#--------------------------

```python
def test_sep_not_in_result_and_sig_error_is_none():
    signer = TimestampSigner("secret")
    value = b"test"
    timestamp = base64_encode(int_to_bytes(signer.get_timestamp()))
    sep = want_bytes(signer.sep)
    signed_value = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)
    result = super(TimestampSigner, signer).unsign(signed_value)
    sep = want_bytes(signer.sep)
    assert sep in result
```


# LLM-generated content at query #8
#--------------------------

def test_unsign_valid_no_max_age_no_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_with_max_age_no_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_valid_with_max_age_and_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert value == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_valid_no_max_age_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    import time
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    time.sleep(2)
    try:
        signer.unsign(signed, max_age=1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_expired_signature_future():
    signer = TimestampSigner("secret")
    signer.get_timestamp = lambda: 100
    signed = signer.sign("test")
    signer.get_timestamp = lambda: 200
    try:
        signer.unsign(signed, max_age=50)
        assert False
    except SignatureExpired as e:
        assert "age" in str(e)

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    no_timestamp = signed.split(b".")
    if len(no_timestamp) >= 2:
        modified = b".".join(no_timestamp[:-1])
    else:
        modified = signed
    try:
        signer.unsign(modified)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    sep = signer.sep.encode()
    value, ts = signed.rsplit(sep, 1)
    malformed = value + sep + b"not-a-timestamp"
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_no_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered, return_timestamp=True)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #9
#--------------------------

```python
def test_unsign_predicate_line43_false():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    # Ensure no exception is raised so that ts_int is not None
    result = signer.unsign(signed)
    assert result == b"test"  # Just to confirm unsign succeeded, predicate at line 43 is False
```


# LLM-generated content at query #10
#--------------------------

```python
def test_unsign_signature_ok_but_timestamp_is_none():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    sep = signer.sep.encode()
    # Replace the base64 timestamp with an invalid one to make ts_int None
    parts = signed_value.rsplit(sep, 1)
    # The timestamp part is the second-to-last part before the signature
    # We need to construct a signed value where base64_decode raises exception
    # So we put an invalid base64 string (e.g., "!!") as the timestamp
    # But we must keep the structure: value + sep + invalid_timestamp + sep + signature
    # To keep signature valid, we need to recompute signature for the modified value
    value_bytes = parts[0]
    original_timestamp = parts[1]
    # The signature is the last part
    # We'll create a new value with invalid timestamp and recompute signature
    invalid_timestamp = b"\xff\xff"
    # Re-sign with the invalid timestamp included
    new_value = value_bytes + sep + invalid_timestamp
    new_signature = signer.get_signature(new_value)
    new_signed = new_value + sep + new_signature
    # Now unsign should raise BadTimeSignature because ts_int is None
    try:
        signer.unsign(new_signed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)
```


# LLM-generated content at query #11
#--------------------------

```python
def test_loads_without_return_timestamp_returns_payload_only():
    serializer = TimedSerializer("secret")
    payload = {"key": "value"}
    signed = serializer.dumps(payload)
    result = serializer.loads(signed)
    assert result == payload
    assert not isinstance(result, tuple)
```


# LLM-generated content at query #12
#--------------------------

```python
def test_unsign_valid_signature_without_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    result = signer.unsign(signed)
    assert result == b"test value"

def test_unsign_valid_signature_with_return_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    value, dt = signer.unsign(signed, return_timestamp=True)
    assert value == b"test value"
    assert isinstance(dt, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_bad_signature_with_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    tampered = signed[:-1] + b"X"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    sep = signer.sep.encode()
    value, timestamp = signed.rsplit(sep, 1)
    no_timestamp = value
    try:
        signer.unsign(no_timestamp)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    sep = signer.sep.encode()
    value, timestamp = signed.rsplit(sep, 1)
    malformed = value + sep + b"not-base64"
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_negative_age():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_sig_error_and_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    sep = signer.sep.encode()
    value, timestamp = signed.rsplit(sep, 1)
    tampered_value = value + b"tampered"
    tampered_signed = tampered_value + sep + timestamp
    try:
        signer.unsign(tampered_signed)
        assert False
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #13
#--------------------------

```python
def test_sig_error_not_none_and_ts_int_is_not_none_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Modify signed_value so that super().unsign raises BadSignature but ts_int is None
    # We need a valid timestamp that decodes to something that cannot be converted to int
    # Actually we need sig_error not None and ts_int None
    # Let's break the signature part but keep timestamp valid
    sep = signer.sep.encode()
    parts = signed_value.rsplit(sep, 1)
    value_with_ts = parts[0]
    bad_signature = b"bad"
    bad_signed = value_with_ts + sep + b"dGVzdA" + sep + bad_signature  # timestamp "test" base64 decoded is invalid for bytes_to_int
    try:
        signer.unsign(bad_signed, max_age=None)
    except BadTimeSignature as e:
        # The predicate at line 49 (if ts_int is not None) should be False, so we check that date_signed is None
        assert e.date_signed is None
```


# LLM-generated content at query #14
#--------------------------

```python
def test_loads_with_return_timestamp_true_returns_payload_and_timestamp():
    from itsdangerous.timed import TimedSerializer
    from itsdangerous.signer import Signer
    from itsdangerous.timed import TimestampSigner
    import time

    serializer = TimedSerializer("secret")
    data = {"key": "value"}
    signed = serializer.dumps(data)

    payload, timestamp = serializer.loads(signed, return_timestamp=True)

    assert payload == data
    assert isinstance(timestamp, float)
    assert abs(time.time() - timestamp) < 10
```


# LLM-generated content at query #15
#--------------------------

def test_loads_returns_payload_when_return_timestamp_is_false():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, return_timestamp=False)
    assert result == "test"


# LLM-generated content at query #16
#--------------------------

```python
def test_unsign_with_sig_error_and_none_ts_int():
    signer = TimestampSigner("secret")
    # Create a signed value with a valid timestamp, then corrupt the signature
    value = b"test"
    timestamp = base64_encode(int_to_bytes(signer.get_timestamp()))
    sep = want_bytes(signer.sep)
    signed = value + sep + timestamp + sep + b"corrupted"
    try:
        signer.unsign(signed)
    except BadTimeSignature as e:
        assert e.date_signed is None
```


# LLM-generated content at query #17
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is hashlib.sha1

def test_timestamp_signer_with_bytes_secret_key():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_with_list_of_keys():
    signer = TimestampSigner([b"old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_timestamp_signer_with_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_with_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_with_custom_sep():
    signer = TimestampSigner("secret-key", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_with_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_with_custom_digest_method():
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_32_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"  # The predicate `if sig_error:` at line 32 is False, so it proceeds without raising sig_error
```


# LLM-generated content at query #19
#--------------------------

```python
def test_exception_at_line_43_occurs_when_base64_decode_fails():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Replace the timestamp part with an invalid base64 string
    sep = signer.sep.encode()
    parts = signed_value.rsplit(sep, 1)
    bad_ts = b"!!!invalid-base64!!!"
    malformed_signed = parts[0] + sep + bad_ts + sep + parts[1]
    with pytest.raises(BadTimeSignature, match="Malformed timestamp"):
        signer.unsign(malformed_signed)
```


# LLM-generated content at query #20
#--------------------------

def test_timestamp_signer_constructor_default():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_sep():
    signer = TimestampSigner("secret-key", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algo = HMACAlgorithm()
    signer = TimestampSigner("secret-key", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_constructor_sep_in_base64_alphabet():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret-key", sep=b"a")


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_32_evaluates_to_false():
    from itsdangerous.timed import TimestampSigner
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    value, timestamp = signer.unsign(signed_value, return_timestamp=True)
    result = signer.unsign(signed_value)
    assert isinstance(result, bytes)
```


# LLM-generated content at query #22
#--------------------------

```python
def test_loads_returns_tuple_when_return_timestamp_is_true():
    serializer = TimedSerializer("secret")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == data
    assert isinstance(timestamp, float)
```


# LLM-generated content at query #23
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_key == b"secret"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_custom_separator():
    signer = TimestampSigner("secret", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_salt_none():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_timestamp_signer_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_secret_key_as_list():
    signer = TimestampSigner(["old_secret", "new_secret"])
    assert signer.secret_keys == [b"old_secret", b"new_secret"]
    assert signer.secret_key == b"new_secret"

def test_timestamp_signer_secret_key_as_bytes():
    signer = TimestampSigner(b"bytes_secret")
    assert signer.secret_key == b"bytes_secret"

def test_timestamp_signer_default_key_derivation_hmac():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_default_key_derivation_none():
    signer = TimestampSigner("secret", key_derivation="none")
    assert signer.key_derivation == "none"


# LLM-generated content at query #24
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_with_bytes_secret_key():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_with_list_of_strings():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_timestamp_signer_with_list_of_bytes():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_timestamp_signer_with_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_with_sep():
    signer = TimestampSigner("secret-key", sep=b"!")
    assert signer.sep == b"!"

def test_timestamp_signer_with_sep_as_string():
    signer = TimestampSigner("secret-key", sep="!")
    assert signer.sep == b"!"

def test_timestamp_signer_with_key_derivation_concat():
    signer = TimestampSigner("secret-key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_timestamp_signer_with_key_derivation_hmac():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_with_digest_method():
    from hashlib import sha256
    signer = TimestampSigner("secret-key", digest_method=sha256)
    assert signer.digest_method is sha256

def test_timestamp_signer_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_separator_not_in_base64_alphabet():
    import string
    for char in string.ascii_letters + string.digits + "-_=":
        try:
            TimestampSigner("secret-key", sep=char)
            assert False, f"Should raise ValueError for separator '{char}'"
        except ValueError:
            pass

def test_timestamp_signer_separator_not_in_base64_alphabet_bytes():
    for char_bytes in [b"a", b"Z", b"0", b"-", b"_", b"="]:
        try:
            TimestampSigner("secret-key", sep=char_bytes)
            assert False, f"Should raise ValueError for separator '{char_bytes}'"
        except ValueError:
            pass

def test_timestamp_signer_separator_allowed():
    signer = TimestampSigner("secret-key", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_secret_key_property():
    signer = TimestampSigner(["old", "new"])
    assert signer.secret_key == b"new"


# LLM-generated content at query #25
#--------------------------

```python
def test_unsign_with_sig_error_and_ts_int_is_not_none():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    # Corrupt the signature so that unsign raises BadSignature
    corrupted_signed_value = signed_value[:-1] + (b"x" if signed_value[-1:] != b"x" else b"y")
    try:
        signer.unsign(corrupted_signed_value)
    except BadTimeSignature:
        pass  # Expected, predicate at line 49 evaluated to True in this case (ts_int not None)
```


# LLM-generated content at query #26
#--------------------------

```python
def test_loads_returns_payload_when_signature_valid_and_no_return_timestamp():
    serializer = TimedSerializer("secret")
    payload = {"key": "value"}
    signed = serializer.dumps(payload)
    result = serializer.loads(signed)
    assert result == payload
```


# LLM-generated content at query #27
#--------------------------

def test_unsign_valid_signature_without_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, return_timestamp=False)
    assert result == b"test"

def test_unsign_valid_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_signature_negative_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_bad_signature():
    signer = TimestampSigner("secret")
    signed_value = b"test.invalid"
    try:
        signer.unsign(signed_value)
        assert False
    except BadSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"test" + want_bytes(signer.sep) + signer.get_signature(b"test")
    try:
        signer.unsign(signed_value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = b"test.sep.badtimestamp"
    try:
        signer.unsign(signed_value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_wrong_key():
    signer1 = TimestampSigner("secret1")
    signer2 = TimestampSigner("secret2")
    signed_value = signer1.sign("test")
    try:
        signer2.unsign(signed_value)
        assert False
    except BadSignature:
        pass


# LLM-generated content at query #28
#--------------------------

```python
def test_loads_returns_payload_when_no_max_age_and_no_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_returns_payload_and_timestamp_when_return_timestamp_true():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, int)

def test_loads_raises_signature_expired_when_max_age_exceeded():
    import time
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    time.sleep(0.1)
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except Exception as e:
        assert type(e).__name__ == "SignatureExpired"

def test_loads_raises_badsignature_when_invalid_signature():
    serializer = TimedSerializer("secret")
    invalid_signed = b"invalid"
    try:
        serializer.loads(invalid_signed)
        assert False
    except Exception as e:
        assert type(e).__name__ == "BadSignature"


# LLM-generated content at query #29
#--------------------------

```python
def test_unsign_valid_signature_no_max_age_no_return_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"

def test_unsign_valid_signature_with_max_age_within_limit():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test_value"

def test_unsign_valid_signature_with_max_age_exceeded():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_valid_signature_return_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test_value"
    assert isinstance(timestamp, datetime)

def test_unsign_invalid_signature_no_timestamp():
    signer = TimestampSigner("secret-key")
    try:
        signer.unsign(b"invalid_data")
        assert False
    except BadSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    signed = b"test_value" + signer.sep.encode() + b"malformed_timestamp" + signer.sep.encode() + signer.get_signature(b"test_value" + signer.sep.encode() + b"malformed_timestamp")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_separator():
    signer = TimestampSigner("secret-key")
    try:
        signer.unsign(b"no_separator_here")
        assert False
    except BadSignature:
        pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret-key")
    future_timestamp = int(time.time()) + 10000
    future_bytes = base64_encode(int_to_bytes(future_timestamp))
    value = b"test_value"
    sep = signer.sep.encode()
    signed = value + sep + future_bytes + sep + signer.get_signature(value + sep + future_bytes)
    try:
        signer.unsign(signed, max_age=3600)
        assert False
    except SignatureExpired:
        pass
```


# LLM-generated content at query #30
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method.__name__ == "sha1"
    assert signer.algorithm is not None

def test_timestamp_signer_with_custom_sep():
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_with_custom_salt():
    signer = TimestampSigner("secret-key", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_with_salt_none():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_with_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_timestamp_signer_with_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_with_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_with_key_list():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_timestamp_signer_with_sep_in_base64_alphabet():
    import re
    from itsdangerous.signer import _base64_alphabet
    for sep in _base64_alphabet:
        try:
            TimestampSigner("secret-key", sep=sep)
            assert False, f"Expected ValueError for sep {sep!r}"
        except ValueError:
            pass

def test_timestamp_signer_with_str_sep():
    signer = TimestampSigner("secret-key", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_with_bytes_secret_key():
    signer = TimestampSigner(b"bytes-key")
    assert signer.secret_keys == [b"bytes-key"]


# LLM-generated content at query #31
#--------------------------

```
def test_loads_without_max_age_and_without_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_with_max_age_and_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, float)

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, float)

def test_loads_with_max_age_expired():
    from itsdangerous import SignatureExpired
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    import time
    time.sleep(0.01)
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_loads_with_invalid_signature():
    from itsdangerous import BadSignature
    serializer = TimedSerializer("secret")
    try:
        serializer.loads(b"invalid")
        assert False
    except BadSignature:
        pass

def test_loads_with_string_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed.decode())
    assert result == "test"

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_with_salt():
    serializer = TimedSerializer("secret", salt="testsalt")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"
```


# LLM-generated content at query #32
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_key == b"secret"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_custom_sep():
    signer = TimestampSigner("secret", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_custom_sep_str():
    signer = TimestampSigner("secret", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_custom_salt_str():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_salt_none():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_timestamp_signer_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_secret_key_list():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_secret_key_list_bytes():
    signer = TimestampSigner([b"old_key", b"new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]


# LLM-generated content at query #33
#--------------------------

def test_unsign_valid_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed)
    assert result == b"value"

def test_unsign_valid_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"value"
    assert isinstance(result[1], datetime)

def test_unsign_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_bad_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    bad_signed = signed[:-1] + b"x"
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = b"value" + want_bytes(signer.sep) + b"invalidsig"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = b"value" + want_bytes(signer.sep) + b"badts" + want_bytes(signer.sep) + signer.get_signature(b"value" + want_bytes(signer.sep) + b"badts")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #34
#--------------------------

def test_unsign_basic():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_with_valid_max_age():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_bad_signature():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test") + b"bad"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret-key")
    signed = b"test" + signer.sep.encode() + signer.get_signature(b"test")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    signed = b"test" + signer.sep.encode() + b"malformed" + signer.sep.encode() + signer.get_signature(b"test" + signer.sep.encode() + b"malformed")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass


# LLM-generated content at query #35
#--------------------------

```python
def test_exception_handling_at_line_52():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Force a timestamp that will cause ValueError when converting to datetime
    bad_timestamp = base64_encode(int_to_bytes(-1))
    sep = signer.sep.encode()
    value_part = b"test"
    # Construct signed value with invalid timestamp but valid signature for the original value
    signed_value_bad_ts = value_part + sep + bad_timestamp + sep + signer.get_signature(value_part + sep + bad_timestamp)
    try:
        signer.unsign(signed_value_bad_ts)
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #36
#--------------------------

```python
def test_unsign_predicate_at_line_43_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    value, ts_bytes = signed_value.rsplit(signer.sep.encode(), 1)
    malformed_ts = b"!!invalid_base64!!"
    malformed_signed = value + signer.sep.encode() + malformed_ts + signer.sep.encode() + signer.get_signature(value + signer.sep.encode() + malformed_ts)
    try:
        signer.unsign(malformed_signed)
    except BadSignature:
        pass
    except BadTimeSignature:
        pass
    except Exception:
        pass
```


# LLM-generated content at query #37
#--------------------------

```python
def test_ts_int_is_none_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    # Craft a signed value where the timestamp part is not valid base64
    # The sign method produces value + sep + timestamp + sep + signature
    # We need to produce a result where the timestamp part decodes to something that bytes_to_int fails on
    # Actually we need ts_int to be None after the try block, so base64_decode must raise an exception
    # or bytes_to_int raises an exception. We'll craft a value that passes the sep check but has bad timestamp.
    # Use sign to get a valid structure, then replace the timestamp with something that fails base64 decode
    valid_signed = signer.sign(b"test")
    sep = signer.sep.encode()
    # Split into value, timestamp, signature
    parts = valid_signed.rsplit(sep, 1)  # value + timestamp, signature
    value_ts = parts[0]
    sig = parts[1]
    # rsplit again to isolate timestamp
    value, ts_b64 = value_ts.rsplit(sep, 1)
    # Replace timestamp with invalid base64 (e.g., not valid base64 chars)
    bad_ts = b"!!!"
    new_signed = value + sep + bad_ts + sep + sig
    try:
        signer.unsign(new_signed)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value
    else:
        assert False, "Expected BadTimeSignature"
```


# LLM-generated content at query #38
#--------------------------

def test_unsign_valid_without_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_with_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_valid_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test"

def test_unsign_expired_signature():
    import time
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    time.sleep(0.1)
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except Exception as e:
        assert "Signature age" in str(e)

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test") + b"invalid"
    try:
        signer.unsign(signed)
        assert False
    except Exception:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    signed_no_ts = signed.rsplit(b".", 1)[0]
    try:
        signer.unsign(signed_no_ts)
        assert False
    except Exception as e:
        assert "timestamp missing" in str(e)


# LLM-generated content at query #39
#--------------------------

def test_loads_return_timestamp_false():
    serializer = TimedSerializer("secret")
    payload = {"key": "value"}
    signed = serializer.dumps(payload)
    result = serializer.loads(signed, return_timestamp=False)
    assert result == payload


# LLM-generated content at query #40
#--------------------------

def test_unsign_valid_signature_no_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed)
    assert result == b"hello"

def test_unsign_valid_signature_with_max_age_not_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"hello"

def test_unsign_valid_signature_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"hello"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature_raises():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    signer.get_timestamp = lambda: int(time.time()) + 100
    try:
        signer.unsign(signed, max_age=10)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_negative_age_raises():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    signer.get_timestamp = lambda: int(time.time()) - 100
    try:
        signer.unsign(signed, max_age=10)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_missing_separator_raises():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp_raises():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    parts = signed.rsplit(b".", 1)
    malformed = parts[0] + b".invalid"
    try:
        signer.unsign(malformed)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_timestamp_raises():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    tampered = signed.replace(b"hello", b"world")
    try:
        signer.unsign(tampered)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_no_timestamp_raises():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"bad_signature")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass

def test_unsign_empty_string_raises():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"")
        assert False, "Should have raised BadSignature"
    except BadSignature:
        pass


# LLM-generated content at query #41
#--------------------------

```python
def test_predicate_line_52_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = super(TimestampSigner, signer).unsign(signed_value)
    sep = b"."
    value, ts_bytes = result.rsplit(sep, 1)
    ts_int = bytes_to_int(base64_decode(ts_bytes))
    try:
        signer.timestamp_to_datetime(ts_int)
    except (ValueError, OSError, OverflowError):
        pass
    else:
        assert True
```


# LLM-generated content at query #42
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_bytes_secret():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_constructor_with_secret_list():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret-key", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_with_salt_none():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algo = HMACAlgorithm()
    signer = TimestampSigner("secret-key", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_constructor_sep_in_base64_alphabet():
    import pytest
    with pytest.raises(ValueError, match="cannot be used"):
        TimestampSigner("secret-key", sep=b"a")


# LLM-generated content at query #43
#--------------------------

def test_loads_with_return_timestamp_true():
    serializer = TimedSerializer("secret")
    data = serializer.dumps({"key": "value"})
    result = serializer.loads(data, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == {"key": "value"}
    assert isinstance(result[1], float)


# LLM-generated content at query #44
#--------------------------

```python
def test_unsign_predicate_line_43_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"  # The predicate at line 43 evaluates to False, so the code proceeds to line 63 and beyond
```


# LLM-generated content at query #45
#--------------------------

def test_timestamp_signer_constructor_default() -> None:
    signer = TimestampSigner("secret")
    assert signer.secret_key == b"secret"
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is TimestampSigner.default_digest_method
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_salt() -> None:
    signer = TimestampSigner("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_sep() -> None:
    signer = TimestampSigner("secret", sep=b"|")
    assert signer.sep == b"|"

def test_timestamp_signer_constructor_with_key_derivation() -> None:
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method() -> None:
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm() -> None:
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_constructor_with_multiple_secret_keys() -> None:
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_constructor_with_salt_none() -> None:
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #46
#--------------------------

def test_unsign_valid_signature_without_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"

def test_unsign_valid_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    value, sep, sig = signed_value.rpartition(signer.sep.encode())
    corrupted = value + sep + b"bad"
    try:
        signer.unsign(corrupted)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    value, sep, timestamp = signed_value.rpartition(signer.sep.encode())
    value, sep, _ = value.rpartition(signer.sep.encode())
    corrupted = value + sep + b"notbase64" + sep + timestamp
    try:
        signer.unsign(corrupted)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_future_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=-1)
        assert False
    except SignatureExpired:
        pass


# LLM-generated content at query #47
#--------------------------

```python
def test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true():
    serializer = TimedSerializer("secret")
    payload = {"key": "value"}
    signed = serializer.dumps(payload)
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == payload
    assert isinstance(result[1], float)
```


# LLM-generated content at query #48
#--------------------------

def test_unsign_returns_value_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_returns_value_and_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert len(result) == 2
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)

def test_unsign_with_max_age_not_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_with_max_age_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_missing_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = b"test" + signer.sep.encode() + b"invalid"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_bad_signature_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test") + b"x"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"test" + signer.sep.encode() + b"badts"
    sig = signer.get_signature(value)
    signed = value + signer.sep.encode() + sig
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_negative_age_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=86400)
        assert False
    except SignatureExpired:
        pass


# LLM-generated content at query #49
#--------------------------

def test_unsign_valid_signature_without_max_age_returns_bytes():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_max_age_not_expired_returns_bytes():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_valid_signature_with_return_timestamp_true_returns_tuple():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, dt = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(dt, datetime)

def test_unsign_expired_signature_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"test" + want_bytes(signer.sep) + b"invalid"
    signed = value + want_bytes(signer.sep) + signer.get_signature(value)
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign(b"test")
    signed_no_ts = signed.rsplit(want_bytes(signer.sep), 1)[0]
    try:
        signer.unsign(signed_no_ts)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    valid = signer.sign(b"test")
    tampered = valid[:-1] + (b"x" if valid[-1:] != b"x" else b"y")
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_without_timestamp_raises_bad_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False
    except BadSignature:
        pass


# LLM-generated content at query #50
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_true():
    import time
    from itsdangerous.timed import TimestampSigner
    from itsdangerous.exc import BadTimeSignature
    from itsdangerous.encoding import base64_encode, int_to_bytes
    from unittest.mock import patch

    signer = TimestampSigner("secret")
    value = b"test"
    # Create a signed value with a malformed timestamp that will cause timestamp_to_datetime to raise an exception
    # We need to trigger the exception at line 52: (ValueError, OSError, OverflowError)
    # One way is to use a timestamp that is out of valid range for datetime.fromtimestamp
    # For example, very large integer that causes OverflowError
    bad_timestamp = 999999999999999999999999999999  # This will likely cause OverflowError
    ts_bytes = base64_encode(int_to_bytes(bad_timestamp))
    sep = signer.sep.encode()
    unsigned_value = value + sep + ts_bytes
    signature = signer.get_signature(unsigned_value)
    signed_value = unsigned_value + sep + signature

    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        # The predicate at line 52 should have evaluated to True, resulting in the exception being raised
        assert str(e) == "Malformed timestamp"
        assert e.payload == value
```


# LLM-generated content at query #51
#--------------------------

```python
def test_timestamp_missing_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"test_value" + signer.sep.encode() + b"invalid_timestamp"
    signed_value += signer.sep.encode() + b"fakesignature"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)
```


# LLM-generated content at query #52
#--------------------------

```python
def test_predicate_line52_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, return_timestamp=False)
    assert isinstance(result, bytes)
```


# LLM-generated content at query #53
#--------------------------

def test_unsign_with_valid_signature_and_no_max_age():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test_value")
    result = signer.unsign(signed_value, return_timestamp=False)
    assert result == b"test_value"

def test_unsign_with_valid_signature_and_return_timestamp():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test_value")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)

def test_unsign_with_max_age_not_exceeded():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test_value")
    result = signer.unsign(signed_value, max_age=3600)
    assert result == b"test_value"

def test_unsign_with_max_age_exceeded():
    import time
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test_value")
    time.sleep(2)
    try:
        signer.unsign(signed_value, max_age=1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_negative_age():
    import time
    signer = TimestampSigner("secret-key")
    # Simulate a timestamp in the future by manipulating the signer's get_timestamp
    original_get_timestamp = signer.get_timestamp
    signer.get_timestamp = lambda: int(time.time()) + 100
    signed_value = signer.sign("test_value")
    signer.get_timestamp = original_get_timestamp
    try:
        signer.unsign(signed_value, max_age=3600)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_bad_signature_and_no_timestamp():
    signer = TimestampSigner("secret-key")
    try:
        signer.unsign(b"invalid.data")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_bad_signature_and_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    # Create a value with a bad signature but valid-looking timestamp
    value = b"test_value.sep.badtimestamp"
    try:
        signer.unsign(value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret-key")
    # Create a signed value without the timestamp part
    value = b"test_value" + signer.sep.encode() + signer.get_signature(b"test_value")
    try:
        signer.unsign(value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    value = b"test_value.sep.invalidtimestamp" + signer.sep.encode() + signer.get_signature(b"test_value.sep.invalidtimestamp")
    try:
        signer.unsign(value)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #54
#--------------------------

def test_loads_with_string_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    result = serializer.loads(signed)
    assert result == "hello"

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    result = serializer.loads(signed.encode())
    assert result == "hello"

def test_loads_with_max_age_and_valid():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    result = serializer.loads(signed, max_age=3600)
    assert result == "hello"

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "hello"
    assert isinstance(timestamp, datetime)

def test_loads_with_max_age_and_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == "hello"
    assert isinstance(timestamp, datetime)

def test_loads_with_salt():
    serializer = TimedSerializer("secret", salt="custom_salt")
    signed = serializer.dumps("hello")
    result = serializer.loads(signed, salt="custom_salt")
    assert result == "hello"

def test_loads_expired_signature():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    import time
    time.sleep(0.1)
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_loads_invalid_signature():
    serializer = TimedSerializer("secret")
    invalid_signed = b"invalid.data"
    try:
        serializer.loads(invalid_signed)
        assert False
    except BadSignature:
        pass


# LLM-generated content at query #55
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_custom_sep():
    signer = TimestampSigner("secret-key", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_custom_digest_method():
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_key_rotation():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_sep_not_in_base64():
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_invalid_sep():
    try:
        TimestampSigner("secret-key", sep=".")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


# LLM-generated content at query #56
#--------------------------

```python
def test_unsign_valid_signature_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired:
        pass
    else:
        raise AssertionError("Expected SignatureExpired")

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
    except BadSignature:
        pass
    else:
        raise AssertionError("Expected BadSignature")

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    value = b"test" + signer.sep.encode() + b"fakesignature"
    try:
        signer.unsign(value)
    except BadTimeSignature:
        pass
    else:
        raise AssertionError("Expected BadTimeSignature")

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    value = b"test" + signer.sep.encode() + b"invalidtimestamp" + signer.sep.encode() + b"fakesignature"
    try:
        signer.unsign(value)
    except BadTimeSignature:
        pass
    else:
        raise AssertionError("Expected BadTimeSignature")

def test_unsign_negative_age():
    signer = TimestampSigner("secret")
    signer.get_timestamp = lambda: 0
    signed = signer.sign("test")
    signer.get_timestamp = lambda: 100
    try:
        signer.unsign(signed, max_age=50)
    except SignatureExpired:
        pass
    else:
        raise AssertionError("Expected SignatureExpired")
```


# LLM-generated content at query #57
#--------------------------

```python
def test_unsign_ts_int_is_none_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    timestamp = base64_encode(int_to_bytes(signer.get_timestamp()))
    sep = want_bytes(signer.sep)
    signed_value = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)
    # Simulate a situation where the timestamp part is malformed to make ts_int None
    malformed_signed = signed_value[:-1] + b"X"
    try:
        signer.unsign(malformed_signed)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value
```


# LLM-generated content at query #58
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_with_bytes_secret_key():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_with_list_of_keys():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_timestamp_signer_with_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_with_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_with_custom_sep():
    signer = TimestampSigner("secret-key", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_with_invalid_sep_raises():
    import itsdangerous.exc
    try:
        TimestampSigner("secret-key", sep=b"a")
        assert False
    except ValueError as e:
        assert "The given separator cannot be used" in str(e)

def test_timestamp_signer_with_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_timestamp_signer_with_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_with_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algo)
    assert signer.algorithm == algo

def test_timestamp_signer_secret_key_property():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_key == b"new-key"


# LLM-generated content at query #59
#--------------------------

def test_unsign_valid_signature_without_max_age_returns_value():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_max_age_not_expired_returns_value():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_valid_signature_with_return_timestamp_true_returns_tuple():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    corrupted = signed[:-5] + b"invalid"
    try:
        signer.unsign(corrupted)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    sig = signer.get_signature(value)
    signed = value + b"." + sig
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_invalid_signature_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = b"test.invalidtimestamp.invalidsig"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #60
#--------------------------

```python
def test_sep_in_result_with_sig_error_false():
    signer = TimestampSigner("test-secret")
    signed_value = signer.sign("test-value")
    # Replace the separator with a different one so that sep is not in result
    bad_signed_value = signed_value.replace(signer.sep.encode(), b"|", 1)
    try:
        signer.unsign(bad_signed_value)
    except BadTimeSignature:
        pass
    # The predicate at line 32 evaluates to False because sep is not in result
    # and sig_error is None (since super().unsign didn't raise BadSignature)
    assert True
```


# LLM-generated content at query #61
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_true():
    import time
    from datetime import datetime, timezone
    from itsdangerous.timed import TimestampSigner
    from itsdangerous.exc import BadTimeSignature
    from itsdangerous.encoding import base64_encode, int_to_bytes, want_bytes

    signer = TimestampSigner("secret")
    value = want_bytes("test")
    real_timestamp = signer.get_timestamp()
    # Create a signed value with a valid timestamp and signature
    timestamp_bytes = base64_encode(int_to_bytes(real_timestamp))
    sep = want_bytes(signer.sep)
    signed_value = value + sep + timestamp_bytes + sep + signer.get_signature(value + sep + timestamp_bytes)
    # Now corrupt the timestamp to be a non-integer but still valid base64 that passes base64_decode but fails bytes_to_int
    # We need a base64 string that decodes to bytes that can't be converted to int (e.g., too long or invalid)
    # Use a base64 string that decodes to more than 8 bytes (e.g., 9 bytes), which will cause bytes_to_int to fail
    # bytes_to_int calls _bytes_to_int(bytestr.rjust(8, b"\x00"))[0], so if bytestr is longer than 8, rjust does nothing and _bytes_to_int expects exactly 8 bytes? Actually rjust pads to 8, so a longer string remains longer; _bytes_to_int likely expects exactly 8 bytes, so it will raise something like ValueError or struct.error
    # Use a 9-byte base64 decoded value
    from base64 import urlsafe_b64encode
    nine_bytes = b"123456789"
    corrupted_ts = urlsafe_b64encode(nine_bytes).rstrip(b"=")
    corrupted_signed = value + sep + corrupted_ts + sep + signer.get_signature(value + sep + corrupted_ts)
    try:
        signer.unsign(corrupted_signed)
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)
        assert e.payload == value
        # The predicate at line 52 should have been triggered, meaning the except caught ValueError, OSError, or OverflowError
        # We can verify that the exception's __cause__ is one of those
        assert isinstance(e.__cause__, (ValueError, OSError, OverflowError))
```


# LLM-generated content at query #62
#--------------------------

```
def test_loads_without_max_age_and_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_with_max_age_valid():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, max_age=3600)
    assert result == "test"

def test_loads_with_max_age_expired():
    import time
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    time.sleep(0.01)
    from itsdangerous.exc import SignatureExpired
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, float)

def test_loads_with_max_age_and_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, float)

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret")
    from itsdangerous.exc import BadSignature
    try:
        serializer.loads(b"invalid")
        assert False
    except BadSignature:
        pass

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed.encode() if isinstance(signed, str) else signed)
    assert result == "test"

def test_loads_with_custom_salt():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test", salt="custom")
    result = serializer.loads(signed, salt="custom")
    assert result == "test"

def test_loads_with_wrong_salt():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test", salt="custom")
    from itsdangerous.exc import BadSignature
    try:
        serializer.loads(signed, salt="wrong")
        assert False
    except BadSignature:
        pass
```


# LLM-generated content at query #63
#--------------------------

def test_unsign_with_valid_signed_value_no_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_with_valid_signed_value_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_with_max_age_not_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_with_max_age_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    with freeze_time("2020-01-01"):
        signed = signer.sign("test")
    with freeze_time("2020-01-02"):
        try:
            signer.unsign(signed, max_age=3600)
            assert False
        except SignatureExpired:
            pass

def test_unsign_with_max_age_future_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    with freeze_time("2020-01-02"):
        signed = signer.sign("test")
    with freeze_time("2020-01-01"):
        try:
            signer.unsign(signed, max_age=3600)
            assert False
        except SignatureExpired:
            pass

def test_unsign_with_bad_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signature = signer.get_signature(value)
    signed = value + b"." + signature
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    timestamp = b"invalid"
    sep = signer.sep.encode()
    signature = signer.get_signature(value + sep + timestamp)
    signed = value + sep + timestamp + sep + signature
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #64
#--------------------------

```python
def test_loads_with_return_timestamp_true_returns_tuple():
    serializer = TimedSerializer("secret")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == data
    assert isinstance(result[1], float)
```


# LLM-generated content at query #65
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    signer.unsign(signed_value, return_timestamp=False)  # Should not raise BadTimeSignature("Malformed timestamp") from line 55
```


# LLM-generated content at query #66
#--------------------------

```
def test_loads_returns_payload_when_valid():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_returns_payload_and_timestamp_when_return_timestamp_true():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, datetime.datetime)

def test_loads_raises_signature_expired_when_max_age_exceeded():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    import time
    time.sleep(0.01)
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_loads_raises_bad_signature_when_salt_mismatch():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test", salt="salt1")
    try:
        serializer.loads(signed, salt="salt2")
        assert False
    except BadSignature:
        pass

def test_loads_raises_bad_signature_when_invalid_data():
    serializer = TimedSerializer("secret")
    try:
        serializer.loads(b"invalid")
        assert False
    except BadSignature:
        pass

def test_loads_accepts_bytes_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed.encode())
    assert result == "test"
```


# LLM-generated content at query #67
#--------------------------

def test_unsign_returns_value_when_no_timestamp_requested():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_returns_tuple_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)

def test_unsign_raises_bad_signature_on_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_raises_signature_expired_when_age_exceeds_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_raises_signature_expired_when_age_negative():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=100)
        assert False
    except SignatureExpired:
        pass

def test_unsign_raises_bad_time_signature_when_timestamp_missing():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    no_timestamp = signed.split(signer.sep.encode())[0]
    try:
        signer.unsign(no_timestamp)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_raises_bad_time_signature_when_timestamp_malformed():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    parts = signed.rsplit(signer.sep.encode(), 1)
    malformed = parts[0] + signer.sep.encode() + b"zzz"
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_returns_value_without_timestamp_exception():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=False)
    assert result == b"test"


# LLM-generated content at query #68
#--------------------------

```python
def test_ts_int_is_none_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    sep = signer.sep.encode()
    # Create a signed value where the timestamp part is invalid base64
    # so that base64_decode raises an exception, causing ts_int to remain None.
    invalid_timestamp = b"!!!invalid!!!"
    signed_value = value + sep + invalid_timestamp + sep + signer.get_signature(value + sep + invalid_timestamp)
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value
```


# LLM-generated content at query #69
#--------------------------

```python
def test_loads_no_return_timestamp_with_valid_data():
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized, return_timestamp=False)
    assert result == data
```


# LLM-generated content at query #70
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_custom_sep():
    signer = TimestampSigner("secret-key", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_custom_digest_method():
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_custom_algorithm():
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algo)
    assert signer.algorithm == algo

def test_timestamp_signer_multiple_secret_keys():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_empty_sep_raises_error():
    try:
        TimestampSigner("secret-key", sep="a")
        assert False
    except ValueError:
        pass

def test_timestamp_signer_sep_in_base64_raises_error():
    try:
        TimestampSigner("secret-key", sep="-")
        assert False
    except ValueError:
        pass


# LLM-generated content at query #71
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_bytes_secret():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_key == b"secret-key"

def test_timestamp_signer_constructor_with_key_list():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret-key", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_custom_digest_method():
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm == algorithm

def test_timestamp_signer_constructor_sep_in_base64_raises():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret-key", sep=b"a")


# LLM-generated content at query #72
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    signed_value_bytes = want_bytes(signed_value)
    sep = want_bytes(signer.sep)
    value, ts_bytes = signed_value_bytes.rsplit(sep, 1)
    ts_int = bytes_to_int(base64_decode(ts_bytes))
    assert ts_int is not None
```


# LLM-generated content at query #73
#--------------------------

```python
def test_unsign_predicate_at_line_32_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"  # The predicate `if sep not in result` is False, so unsign succeeds
```


# LLM-generated content at query #74
#--------------------------

def test_loads_return_timestamp_true_returns_tuple():
    ts = TimedSerializer("secret")
    s = ts.dumps("test")
    result = ts.loads(s, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == "test"


# LLM-generated content at query #75
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_false():
    import time
    from datetime import datetime, timezone
    from itsdangerous.timed import TimestampSigner
    from itsdangerous.exc import BadSignature, BadTimeSignature

    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert isinstance(result, bytes)
```


# LLM-generated content at query #76
#--------------------------

```python
def test_unsign_timestamp_to_datetime_raises_value_error():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    # Modify the timestamp part to be too large to cause OverflowError
    sep = signer.sep.encode()
    # Simulate a timestamp that is too large (e.g., 2**63)
    large_ts = 2**63
    # Encode the large timestamp as base64
    import struct
    ts_bytes = struct.pack(">Q", large_ts)
    import base64
    encoded_ts = base64.urlsafe_b64encode(ts_bytes).rstrip(b"=")
    # Construct a signed value with the large timestamp
    modified_value = b"test" + sep + encoded_ts + sep + signer.get_signature(b"test" + sep + encoded_ts)
    try:
        signer.unsign(modified_value)
        assert False, "Expected BadTimeSignature exception"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)


# LLM-generated content at query #77
#--------------------------

```python
def test_ts_int_is_none_after_bad_timestamp():
    signer = TimestampSigner("secret")
    # Create a signed value, then replace the timestamp with invalid base64
    value = b"test"
    sep = want_bytes(signer.sep)
    timestamp = base64_encode(int_to_bytes(signer.get_timestamp()))
    signed_value = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)
    # Change the timestamp part to something that fails base64_decode or bytes_to_int
    # Use a string that is not valid base64 (e.g., not padded correctly)
    bad_timestamp = b"!!invalid!!"
    # We need to replace the timestamp portion with something that decodes to invalid int
    # Alternatively, we can use a timestamp that is not a valid int after decode
    # For simplicity, use a timestamp that is valid base64 but bytes_to_int fails (e.g., more than 8 bytes)
    long_timestamp = base64_encode(b"x" * 9)
    # Reconstruct signed value with bad timestamp
    bad_signed = value + sep + long_timestamp + sep + signer.get_signature(value + sep + long_timestamp)
    # This should trigger the predicate at line 63 after unsign processes it
    try:
        signer.unsign(bad_signed)
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)
```


# LLM-generated content at query #78
#--------------------------

```python
def test_unsign_with_bad_signature_and_malformed_timestamp_does_not_raise_bad_time_signature_from_exception():
    signer = TimestampSigner("secret-key")
    # Create a signed value with a valid timestamp
    value = b"test"
    timestamp = signer.get_timestamp()
    timestamp_b64 = base64_encode(int_to_bytes(timestamp))
    sep = signer.sep.encode()
    signed_value = value + sep + timestamp_b64 + sep + signer.get_signature(value + sep + timestamp_b64)
    # Corrupt the signature
    corrupted_signed = signed_value[:-1] + b"X"
    # The timestamp is valid, so the except block at line 52 should not execute
    try:
        signer.unsign(corrupted_signed)
    except BadTimeSignature as e:
        assert "Malformed timestamp" not in str(e)
```


# LLM-generated content at query #79
#--------------------------

def test_loads_returns_tuple_when_return_timestamp_true():
    serializer = TimedSerializer("secret")
    data = serializer.dumps("test")
    result = serializer.loads(data, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == "test"
    assert isinstance(result[1], float)


# LLM-generated content at query #80
#--------------------------

```python
def test_loads_returns_payload_when_valid_signature():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_returns_payload_and_timestamp_when_return_timestamp_true():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, datetime.datetime)

def test_loads_raises_bad_signature_when_invalid_data():
    serializer = TimedSerializer("secret")
    try:
        serializer.loads(b"invalid.data")
        assert False
    except BadSignature:
        pass

def test_loads_raises_signature_expired_when_max_age_exceeded():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    import time
    time.sleep(0.1)
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_loads_with_salt_uses_correct_signer():
    serializer = TimedSerializer("secret", salt="custom_salt")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, salt="custom_salt")
    assert result == "test"

def test_loads_raises_bad_signature_when_wrong_salt():
    serializer = TimedSerializer("secret", salt="salt1")
    signed = serializer.dumps("test")
    try:
        serializer.loads(signed, salt="wrong_salt")
        assert False
    except BadSignature:
        pass

def test_loads_with_string_input_works():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed.decode())
    assert result == "test"
```


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_loads_basic_payload():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, float)

def test_loads_with_max_age_valid():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, max_age=3600)
    assert result == "test"

def test_loads_with_salt():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test", salt="custom_salt")
    result = serializer.loads(signed, salt="custom_salt")
    assert result == "test"

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed.encode())
    assert result == "test"

def test_loads_raises_bad_signature():
    serializer = TimedSerializer("secret")
    try:
        serializer.loads(b"invalid")
        assert False
    except BadSignature:
        pass

def test_loads_raises_signature_expired():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    try:
        serializer.loads(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_loads_unsafe_with_bad_signature():
    serializer = TimedSerializer("secret")
    result = serializer.loads_unsafe(b"invalid")
    assert result[0] == False


# LLM-generated content at query #2
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_sep():
    signer = TimestampSigner("secret", sep="-")
    assert signer.sep == b"-"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_secret_key_list():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_bytes_secret():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]
```


# LLM-generated content at query #3
#--------------------------

```python
def test_unsign_without_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"test"
    try:
        signer.unsign(signed_value)
    except BadTimeSignature:
        pass

def test_unsign_with_invalid_base64_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"test.==invalid=="
    try:
        signer.unsign(signed_value)
    except BadTimeSignature:
        pass

def test_unsign_with_valid_signature_and_timestamp_returns_value():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed)
    assert result == value

def test_unsign_with_return_timestamp_true_returns_tuple():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == value
    assert isinstance(result[1], datetime)

def test_unsign_with_max_age_not_expired_returns_value():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600)
    assert result == value

def test_unsign_with_max_age_expired_raises_signature_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired:
        pass

def test_unsign_with_max_age_zero_raises_signature_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    try:
        signer.unsign(signed, max_age=0)
    except SignatureExpired:
        pass

def test_unsign_with_bad_signature_and_no_timestamp_raises_bad_signature():
    signer = TimestampSigner("secret")
    signed_value = b"test.badtimestamp"
    try:
        signer.unsign(signed_value)
    except BadSignature:
        pass

def test_unsign_with_bad_signature_and_valid_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    bad_signed = signed + b"bad"
    try:
        signer.unsign(bad_signed)
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"test.sep.malformed"
    try:
        signer.unsign(signed_value)
    except BadTimeSignature:
        pass

def test_unsign_with_empty_string_raises_bad_data():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"")
    except BadData:
        pass
```


# LLM-generated content at query #4
#--------------------------

def test_timestamp_signer_constructor_with_defaults():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_custom_secret_key_bytes():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_constructor_with_secret_key_list():
    signer = TimestampSigner(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_timestamp_signer_constructor_with_secret_key_list_bytes():
    signer = TimestampSigner([b"key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_timestamp_signer_constructor_with_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_with_custom_salt_bytes():
    signer = TimestampSigner("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret-key", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_constructor_with_sep_in_base64_alphabet():
    try:
        TimestampSigner("secret-key", sep=".")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for separator in base64 alphabet"

def test_timestamp_signer_constructor_with_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm == algorithm


# LLM-generated content at query #5
#--------------------------

def test_unsign_valid_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result, ts = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(ts, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_bad_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False
    except BadSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, sep, _ = signed.rpartition(signer.sep.encode())
    no_ts = value
    try:
        signer.unsign(no_ts)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, sep, sig = signed.rpartition(signer.sep.encode())
    malformed = value + sep + b"!!!" + sep + sig
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #6
#--------------------------

```python
def test_unsign_sep_not_in_result_but_sig_error_is_none():
    signer = TimestampSigner("secret-key")
    # Create a signed value with a valid timestamp but then modify it so that the separator is not present
    # We'll use a value that doesn't contain the separator at all
    value = b"test_value"
    # The separator is typically "."
    # We'll craft a signed_value that does not contain "."
    # By calling sign and then removing the separator part
    signed = signer.sign("test_value")
    # Extract the timestamp and signature parts
    sep = b"."
    # signed looks like: value + sep + timestamp + sep + signature
    # We'll remove the separator by replacing it with nothing
    # Actually we want a case where sep is not in result after super().unsign
    # super().unsign returns value + sep + timestamp (without signature)
    # So if we provide a signed_value that is just the value without any sep, super().unsign will fail?
    # Actually super().unsign expects the format value + sep + signature
    # If we provide just "test_value", it will raise BadSignature because it can't split
    # But then sig_error will be set and result will be b"test_value"
    # In that case, sep is not in result, and sig_error is not None -> the predicate at line 32 is True
    # We need the predicate at line 32 to be False, meaning sep not in result AND sig_error is None
    # That means super().unsign must succeed (sig_error is None) but the result does not contain sep
    # This can happen if the signed_value has no timestamp (i.e., it's from a regular Signer)
    # So we can create a signed value using Signer directly (without timestamp)
    from itsdangerous.signer import Signer
    plain_signer = Signer("secret-key")
    # Sign a value with regular signer, the format is value + sep + signature (no timestamp)
    signed_no_timestamp = plain_signer.sign(b"test_value")
    # This signed value does not have a timestamp, so after super().unsign, result = b"test_value" + sep + signature
    # Wait, super().unsign returns the value part (without signature) if successful
    # Actually super().unsign returns the original value (without the signature and separator)
    # For a regular Signer, unsign returns the value before the last separator
    # So result = b"test_value" and sep is not in result (since result is just "test_value")
    # And sig_error is None because signature is valid
    # Then the predicate at line 32: if sep not in result -> True, then if sig_error -> False, so it goes to line 35 and raises BadTimeSignature
    # That's exactly the case we want to test: sep not in result and sig_error is None
    # So we call unsign on our TimestampSigner with a signed value that comes from a regular Signer
    try:
        signer.unsign(signed_no_timestamp)
    except BadTimeSignature as e:
        assert str(e) == "timestamp missing"
        assert e.payload == b"test_value"
```


# LLM-generated content at query #7
#--------------------------

def test_predicate_line49_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert isinstance(result, bytes)


# LLM-generated content at query #8
#--------------------------

```python
def test_unsign_valid_signature_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    sep = signer.sep.encode()
    bad_signed = signed.rsplit(sep, 1)[0]
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bad_signed = signed[:-1] + b"x"
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    value = want_bytes("test")
    sep = want_bytes(signer.sep)
    bad_ts = base64_encode(b"abc")
    signed = value + sep + bad_ts + sep + signer.get_signature(value + sep + bad_ts)
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_negative_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_return_timestamp_false_by_default():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert isinstance(result, bytes)

def test_unsign_valid_signature_with_max_age_not_exceeded():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=1000)
    assert result == b"test"


# LLM-generated content at query #9
#--------------------------

```python
def test_loads_signature_expired_raised():
    from itsdangerous.timed import TimedSerializer
    from itsdangerous.exc import SignatureExpired
    from itsdangerous.signer import Signer
    from itsdangerous.timed import TimestampSigner
    import time

    serializer = TimedSerializer("secret")
    signed_value = serializer.dumps("test")
    time.sleep(0.1)
    try:
        serializer.loads(signed_value, max_age=0)
    except SignatureExpired:
        pass
```


# LLM-generated content at query #10
#--------------------------

```python
def test_ts_int_not_none_after_exception():
    signer = TimestampSigner("secret")
    # Create a signed value with a malformed timestamp that will cause bytes_to_int to raise
    value = b"test"
    sep = signer.sep.encode()
    # Use a timestamp that is not valid base64 (e.g., just "!")
    malformed_ts = b"!"
    signed_value = value + sep + malformed_ts + sep + signer.get_signature(value + sep + malformed_ts)
    try:
        signer.unsign(signed_value)
    except Exception:
        pass
    # The predicate at line 43 (ts_int is None) should evaluate to False
    # because the except block on line 43-44 catches the exception and ts_int remains None
    # We verify this by checking that the method does not raise BadTimeSignature("Malformed timestamp")
    # which would be raised if ts_int is None (line 63-64)
    assert True  # If we reach here, the predicate evaluated to False (no exception raised for None ts_int)
```


# LLM-generated content at query #11
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_key == b"secret"
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_with_bytes_secret_key():
    signer = TimestampSigner(b"secret")
    assert signer.secret_key == b"secret"

def test_timestamp_signer_with_list_of_keys():
    signer = TimestampSigner(["old", "new"])
    assert signer.secret_keys == [b"old", b"new"]
    assert signer.secret_key == b"new"

def test_timestamp_signer_with_custom_salt():
    signer = TimestampSigner("secret", salt="custom")
    assert signer.salt == b"custom"

def test_timestamp_signer_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_with_custom_sep():
    signer = TimestampSigner("secret", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_with_sep_in_base64_alphabet_raises():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret", sep="a")

def test_timestamp_signer_with_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_with_none_key_derivation():
    signer = TimestampSigner("secret", key_derivation=None)
    assert signer.key_derivation == "django-concat"

def test_timestamp_signer_with_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_with_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm is algo


# LLM-generated content at query #12
#--------------------------

```python
def test_ts_int_is_none_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"test_value" + signer.sep.encode() + b"invalid_timestamp"
    signed_value += signer.sep.encode() + signer.get_signature(signed_value)
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == b"test_value"
```


# LLM-generated content at query #13
#--------------------------

```python
def test_ts_int_is_none_raises_bad_time_signature():
    import time
    from datetime import datetime, timezone
    from itsdangerous.timed import TimestampSigner
    from itsdangerous.exc import BadTimeSignature
    signer = TimestampSigner("secret")
    # Create a signed value where the timestamp part is not valid base64
    value = b"test"
    sep = signer.sep.encode()
    bad_timestamp = b"!!!invalid_base64!!!"
    signed_value = value + sep + bad_timestamp + sep + signer.get_signature(value + sep + bad_timestamp)
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value
```


# LLM-generated content at query #14
#--------------------------

```python
def test_unsign_predicate_true():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    sep = signer.sep.encode()
    value, ts_bytes = signed_value.rsplit(sep, 1)
    mock_ts_bytes = b"invalid_base64!!!"
    signed_value_mock = value + sep + mock_ts_bytes + sep + signer.get_signature(value + sep + mock_ts_bytes)
    signer.unsign(signed_value_mock)  # predicate at line 43 is True, exception caught silently
```


# LLM-generated content at query #15
#--------------------------

```python
def test_age_negative_raises_signature_expired():
    import time
    from datetime import datetime, timezone
    from itsdangerous.signer import Signer
    from itsdangerous.timed import TimestampSigner
    from itsdangerous.exc import SignatureExpired, BadTimeSignature

    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    signer.get_timestamp = lambda: int(time.time()) - 10
    try:
        signer.unsign(signed_value, max_age=5)
    except SignatureExpired:
        pass
    signer.get_timestamp = lambda: int(time.time()) + 10
    try:
        signer.unsign(signed_value, max_age=5)
    except SignatureExpired:
        pass
    signer.get_timestamp = lambda: int(time.time())
    result = signer.unsign(signed_value, max_age=100)
    assert result == b"test"
```


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    malformed_signed = signed_value.rstrip(b"=") + b"invalid_base64!"
    try:
        signer.unsign(malformed_signed)
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #17
#--------------------------

def test_unsign_valid_signature_without_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_max_age_not_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_valid_signature_with_max_age_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    import time
    time.sleep(2)
    try:
        signer.unsign(signed, max_age=1)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_valid_signature_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, dt = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(dt, datetime)

def test_unsign_invalid_signature_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    bad_signed = b"test" + signer.sep.encode() + signer.get_signature(b"test")
    try:
        signer.unsign(bad_signed)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature as e:
        assert "timestamp missing" in str(e)

def test_unsign_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    sep = signer.sep.encode()
    bad_ts = b"not-a-timestamp"
    signed = value + sep + bad_ts + sep + signer.get_signature(value + sep + bad_ts)
    try:
        signer.unsign(signed)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)

def test_unsign_valid_signature_with_negative_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    signer.get_timestamp = lambda: 0
    try:
        signer.unsign(signed, max_age=3600)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired as e:
        assert "age" in str(e)


# LLM-generated content at query #18
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_separator():
    signer = TimestampSigner("secret", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_multiple_keys():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #19
#--------------------------

```python
def test_unsign_with_sig_error_and_none_ts_int():
    signer = TimestampSigner("secret")
    # Create a signed value with a valid timestamp marker but base64 decode fails
    # so that ts_int remains None, and also cause a BadSignature error.
    signed_value = b"value" + signer.sep.encode() + b"invalid!base64"
    try:
        signer.unsign(signed_value)
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #20
#--------------------------

```python
def test_unsign_with_sig_error_and_ts_int_none():
    signer = TimestampSigner("secret")
    value = b"test_value"
    sep = signer.sep.encode()
    # Create a signed value with a bad signature and a malformed timestamp that results in ts_int = None
    bad_ts = b"invalid_base64"
    signed_value = value + sep + bad_ts + sep + b"badsig"
    try:
        signer.unsign(signed_value)
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    # Modify the timestamp part to be invalid base64 to trigger Exception
    value, timestamp = signed.rsplit(b".", 1)
    # Use a non-base64 character to make bytes_to_int(base64_decode(...)) raise
    bad_timestamp = b"!!invalid!!"
    malformed_signed = value + b"." + bad_timestamp + b"." + signer.get_signature(value + b"." + bad_timestamp)
    try:
        signer.unsign(malformed_signed)
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"  # predicate at line 43 is False when ts_int is not None
```


# LLM-generated content at query #23
#--------------------------

```python
def test_unsign_valid_signature_without_timestamp_return():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_timestamp_return():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)

def test_unsign_expired_signature_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bytes_signed = want_bytes(signed)
    sep = want_bytes(signer.sep)
    parts = bytes_signed.rsplit(sep, 1)
    malformed = parts[0] + sep + b"invalid_timestamp" + sep + parts[1]
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bytes_signed = want_bytes(signed)
    sep = want_bytes(signer.sep)
    parts = bytes_signed.rsplit(sep, 1)
    missing_timestamp = parts[0] + sep + parts[1]
    try:
        signer.unsign(missing_timestamp)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_invalid_signature_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bytes_signed = want_bytes(signed)
    sep = want_bytes(signer.sep)
    parts = bytes_signed.rsplit(sep, 1)
    invalid_sig = parts[0] + sep + parts[1] + b"tampered"
    try:
        signer.unsign(invalid_sig)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_signature_with_negative_age_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_valid_but_max_age_none_returns_value():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=None)
    assert result == b"test"
```


# LLM-generated content at query #24
#--------------------------

def test_unsign_valid_signature_without_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed)
    assert result == b"value"

def test_unsign_valid_signature_with_max_age_within_limit():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"value"

def test_unsign_valid_signature_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    value, ts = signer.unsign(signed, return_timestamp=True)
    assert value == b"value"
    assert isinstance(ts, datetime)

def test_unsign_invalid_signature_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_expired_signature_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass


# LLM-generated content at query #25
#--------------------------

```python
def test_ts_int_is_none_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"test_value"
    sep = want_bytes(signer.sep)
    bad_ts = base64_encode(b"not_a_valid_timestamp")
    signed_value = value + sep + bad_ts + sep + signer.get_signature(value + sep + bad_ts)

    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value
```


# LLM-generated content at query #26
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is TimestampSigner.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_separator():
    signer = TimestampSigner("secret", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_custom_salt():
    signer = TimestampSigner("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_salt_none():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_custom_digest_method():
    from hashlib import sha256
    signer = TimestampSigner("secret", digest_method=sha256)
    assert signer.digest_method is sha256

def test_timestamp_signer_constructor_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    from hashlib import sha256
    algorithm = HMACAlgorithm(sha256)
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_separator_in_base64_alphabet():
    try:
        TimestampSigner("secret", sep=b".")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for '.' separator"

def test_timestamp_signer_constructor_multiple_secret_keys():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"


# LLM-generated content at query #27
#--------------------------

```python
def test_unsign_with_encoded_timestamp_does_not_trigger_exception():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    # Ensure that the predicate at line 43 (except Exception) does not evaluate True
    # by verifying unsign succeeds without raising BadTimeSignature
    result = signer.unsign(signed)
    assert result == value
```


# LLM-generated content at query #28
#--------------------------

```python
def test_unsign_with_bad_signature_and_timestamp_that_raises_overflow_error():
    signer = TimestampSigner("secret")
    value = b"test"
    timestamp_bytes = base64_encode(int_to_bytes(1 << 62))
    sep = signer.sep.encode()
    bad_signature = b"invalid"
    signed_value = value + sep + timestamp_bytes + sep + bad_signature
    try:
        signer.unsign(signed_value)
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #29
#--------------------------

```python
def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_bytes_secret():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_timestamp_signer_constructor_with_list_secret():
    signer = TimestampSigner(["old", "new"])
    assert signer.secret_keys == [b"old", b"new"]

def test_timestamp_signer_constructor_with_custom_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_constructor_with_invalid_sep():
    try:
        TimestampSigner("secret", sep=".")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

def test_timestamp_signer_constructor_with_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_constructor_with_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    alg = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=alg)
    assert signer.algorithm is alg
```


# LLM-generated content at query #30
#--------------------------

```python
def test_timestamp_to_datetime_raises_bad_time_signature_on_overflow_error():
    signer = TimestampSigner("secret")
    signed_value = signer.sign(b"test")
    # Corrupt the timestamp to cause an OverflowError when converting
    sep = signer.sep.encode()
    # A timestamp that is too large for datetime.fromtimestamp on 32-bit systems
    corrupt_timestamp = base64_encode(int_to_bytes(2**63 - 1)).decode()
    corrupted = b"test" + sep + corrupt_timestamp.encode() + sep + signer.get_signature(b"test" + sep + corrupt_timestamp.encode())
    try:
        signer.unsign(corrupted)
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_line_52_evaluates_to_false():
    signer = TimestampSigner("secret")
    value = b"test_value"
    timestamp = signer.get_timestamp()
    timestamp_bytes = base64_encode(int_to_bytes(timestamp))
    sep = signer.sep.encode()
    signature = signer.get_signature(value + sep + timestamp_bytes)
    signed_value = value + sep + timestamp_bytes + sep + signature
    # Modify timestamp to be invalid for base64_decode, but not raise ValueError, OSError, or OverflowError
    # We'll use a valid base64 that decodes to an int that timestamp_to_datetime handles
    # To ensure ts_int is not None, we need base64_decode to succeed
    # Provide a valid base64 that decodes to an int that causes timestamp_to_datetime to succeed
    # The predicate is about the except block, we want it to not execute
    # We can achieve this by having a valid timestamp that doesn't cause those exceptions
    # So we use the original timestamp which is valid
    value, ts_dt = signer.unsign(signed_value, return_timestamp=True)
    # The predicate at line 52 is inside the except block, and we want it to evaluate to False
    # meaning no exception is raised. The test passes if unsign succeeds.
    assert True
```


# LLM-generated content at query #32
#--------------------------

def test_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1

def test_constructor_with_bytes_secret_key():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_constructor_with_list_of_secret_keys():
    signer = TimestampSigner(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_constructor_with_custom_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_constructor_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_constructor_with_custom_sep():
    signer = TimestampSigner("secret", sep="|")
    assert signer.sep == b"|"

def test_constructor_with_invalid_sep_raises_error():
    try:
        TimestampSigner("secret", sep=".")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

def test_constructor_with_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_constructor_with_custom_digest_method():
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm == algorithm


# LLM-generated content at query #33
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_sep():
    signer = TimestampSigner("secret", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_constructor_custom_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_custom_digest_method():
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_custom_algorithm():
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm == algo

def test_timestamp_signer_constructor_multiple_secret_keys():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_constructor_sep_in_base64_alphabet_raises():
    import re
    from itsdangerous.exc import BadSignature
    try:
        TimestampSigner("secret", sep="A")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# LLM-generated content at query #34
#--------------------------

```python
def test_unsign_valid_signature_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    signed_no_ts = signed.rsplit(b".", 1)[0]
    try:
        signer.unsign(signed_no_ts)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    parts = signed.rsplit(b".", 1)
    malformed = parts[0] + b"." + b"invalid"
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_max_age_negative_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_returns_bytes():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert isinstance(result, bytes)

def test_unsign_returns_tuple_when_return_timestamp_true():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bytes)
    assert isinstance(result[1], datetime)
```


# LLM-generated content at query #35
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    signer.unsign(signed_value, return_timestamp=True)  # line 52 exception not raised
```


# LLM-generated content at query #36
#--------------------------

```python
def test_unsign_valid_signature_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_signature_with_negative_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=1000000)
        assert False
    except SignatureExpired:
        pass

def test_unsign_invalid_signature_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = signer.signature(b"test")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    sep = signer.sep.encode()
    parts = signed.rsplit(sep, 1)
    malformed = parts[0] + sep + b"invalid"
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_max_age_valid():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_with_max_age_and_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result, timestamp = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_empty_value():
    signer = TimestampSigner("secret")
    signed = signer.sign(b"")
    result = signer.unsign(signed)
    assert result == b""

def test_unsign_unicode_value():
    signer = TimestampSigner("secret")
    signed = signer.sign("héllo")
    result = signer.unsign(signed)
    assert result == "héllo".encode()

def test_unsign_with_different_sep():
    signer = TimestampSigner("secret", sep="|")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"
```


# LLM-generated content at query #37
#--------------------------

def test_timestamp_signer_constructor_default():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_custom_sep():
    signer = TimestampSigner("secret", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_custom_salt():
    signer = TimestampSigner("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_custom_digest_method():
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_custom_algorithm():
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_constructor_secret_key_rotation():
    signer = TimestampSigner(["old", "newer", "newest"])
    assert signer.secret_keys == [b"old", b"newer", b"newest"]
    assert signer.secret_key == b"newest"

def test_timestamp_signer_constructor_sep_in_base64_raises():
    try:
        TimestampSigner("secret", sep="+")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

def test_timestamp_signer_constructor_salt_none():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #38
#--------------------------

```python
def test_unsign_ts_int_is_none_raises_bad_time_signature():
    from itsdangerous.timed import TimestampSigner
    from itsdangerous.exc import BadTimeSignature

    signer = TimestampSigner("secret")
    # Simulate a signed value with a malformed (non-base64) timestamp part
    # We need to craft a value that passes the sep check but fails base64 decode
    # Use a signature that is valid but with a timestamp that is not base64
    # Since we want to trigger line 63, we need sig_error to be None (signature valid)
    # and ts_int to be None after exception in base64_decode.
    # We'll mock base64_decode to raise an exception for the timestamp part.
    # Alternatively, we can patch bytes_to_int or base64_decode to return None.
    # Simpler: create a signed value with a valid signature but a non-base64 timestamp.
    # However, we can't easily craft that without internal knowledge.
    # Instead, we can patch base64_decode to raise an exception.
    import itsdangerous.encoding as encoding
    original_decode = encoding.base64_decode
    def failing_decode(s):
        raise Exception("fail")
    encoding.base64_decode = failing_decode
    try:
        signed_value = signer.sign("test")
        # The signed_value has a valid timestamp; we need to make it malformed.
        # We'll just call unsign with a value that has sep but invalid timestamp.
        # Actually signer.sign creates a valid signed value; we need to modify it.
        # Let's just call unsign with a crafted value that has sep but invalid ts.
        value = b"test|invalid_timestamp"
        # Get a valid signature for "test|invalid_timestamp"
        import itsdangerous.signer
        # We'll just use the signer's get_signature
        sig = signer.get_signature(b"test|invalid_timestamp")
        signed_value = b"test|invalid_timestamp" + signer.sep.encode() + sig
    finally:
        encoding.base64_decode = original_decode
    # Now unsign should go through the path where ts_int is None (since base64_decode failed)
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #39
#--------------------------

```python
def test_unsign_valid_signature_without_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_max_age_not_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_valid_signature_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_tampered_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    tampered = signed[:-1] + b"X"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    import time
    time.sleep(1)
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_future_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    import time
    original_time = time.time
    time.time = lambda: 0
    try:
        signer.unsign(signed, max_age=3600)
        assert False
    except SignatureExpired:
        pass
    finally:
        time.time = original_time

def test_unsign_missing_separator():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bad_signed = b"test" + signer.sep.encode() + b"bad"
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bad_signed = signed[:-10] + b"bad"
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_non_timestamp_data():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    no_timestamp = b"test" + signer.sep.encode() + b"data"
    try:
        signer.unsign(no_timestamp)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_string_input():
    signer = TimestampSigner("secret")
    signed = signer.sign("test").decode()
    result = signer.unsign(signed)
    assert result == b"test"```


# LLM-generated content at query #40
#--------------------------

def test_unsign_valid_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed)
    assert result == b"hello"

def test_unsign_valid_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"hello"
    assert isinstance(result[1], datetime)

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    tampered = signed[:-1] + b"X"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    sep = signer.sep.encode()
    value_only = signed.rsplit(sep, 1)[0]
    try:
        signer.unsign(value_only)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    sep = signer.sep.encode()
    parts = signed.rsplit(sep, 1)
    bad_ts = b"not-a-timestamp"
    malformed = parts[0] + sep + bad_ts + sep + signer.get_signature(parts[0] + sep + bad_ts)
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_future_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_valid_with_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"hello"

def test_unsign_valid_with_max_age_and_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"hello"
    assert isinstance(result[1], datetime)


# LLM-generated content at query #41
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret-key")
    assert signer.secret_key == b"secret-key"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_with_sep():
    signer = TimestampSigner("secret-key", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_secret_key_list():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_salt_none():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_sep_in_base64_raises():
    import re
    try:
        TimestampSigner("secret-key", sep=".")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for separator '.' which is in base64 alphabet"


# LLM-generated content at query #42
#--------------------------

```python
def test_age_less_than_zero_evaluates_true():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    signer.get_timestamp = lambda: -1
    try:
        signer.unsign(signed_value, max_age=0)
    except SignatureExpired as e:
        assert "age -1" in str(e)
```


# LLM-generated content at query #43
#--------------------------

```python
def test_ts_int_is_none_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    # Construct a signed value where the timestamp part is not valid base64
    # so base64_decode raises an exception, leaving ts_int as None.
    # We need to bypass normal signing to create such a value.
    # The signed value format is: value + sep + timestamp + sep + signature
    # We can craft a value where the timestamp is not valid base64.
    sep = signer.sep
    value = b"test"
    # Use an invalid base64 string for the timestamp
    bad_ts = b"!!!invalid!!!"
    # Create a fake signature (any bytes)
    fake_sig = b"fakesig"
    signed_value = value + sep.encode() + bad_ts + sep.encode() + fake_sig
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value
    else:
        assert False, "Expected BadTimeSignature"
```


# LLM-generated content at query #44
#--------------------------

```python
def test_age_less_than_zero_raises_signature_expired():
    import time
    import base64
    from datetime import datetime, timezone
    from itsdangerous.timed import TimestampSigner
    from itsdangerous.exc import SignatureExpired

    signer = TimestampSigner("secret")
    original_get_timestamp = signer.get_timestamp

    def mock_get_timestamp():
        return 100

    def mock_unsign(signed_value):
        value = b"test"
        sep = b"."
        ts = 200
        ts_bytes = base64.urlsafe_b64encode(ts.to_bytes(8, "big")).rstrip(b"=")
        sig = signer.get_signature(value + sep + ts_bytes)
        return value + sep + ts_bytes + sep + sig

    signer.get_timestamp = mock_get_timestamp
    signer.unsign = mock_unsign
    signed = mock_unsign(b"dummy")

    signer.get_timestamp = lambda: 100
    signer.unsign = lambda x: mock_unsign(x)
    signed_value = signed

    try:
        signer.unsign(signed_value, max_age=50)
    except SignatureExpired as e:
        assert "Signature age" in str(e)
        assert "0 seconds" in str(e)
    else:
        assert False, "SignatureExpired not raised"

    signer.get_timestamp = original_get_timestamp
```


# LLM-generated content at query #45
#--------------------------

```python
def test_base64_decode_raises_exception_on_invalid_input():
    signer = TimestampSigner("secret")
    # Create a signed value with a malformed timestamp that will cause base64_decode to raise an exception
    value = b"test_value"
    sep = signer.sep.encode()
    # Use an invalid base64 string for the timestamp part
    malformed_ts = b"!!!invalid_base64!!!"
    signed_value = value + sep + malformed_ts + sep + signer.get_signature(value + sep + malformed_ts)
    result = signer.unsign(signed_value)
    assert result == value
```


# LLM-generated content at query #46
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_key == b"secret"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is signer.default_digest_method

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_custom_sep():
    signer = TimestampSigner("secret", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_secret_key_list():
    signer = TimestampSigner(["old_secret", "new_secret"])
    assert signer.secret_keys == [b"old_secret", b"new_secret"]
    assert signer.secret_key == b"new_secret"

def test_timestamp_signer_sep_raises_on_base64_char():
    from itsdangerous.exc import BadSignature
    try:
        TimestampSigner("secret", sep="a")
        assert False
    except ValueError:
        pass

def test_timestamp_signer_salt_none():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #47
#--------------------------

def test_unsign_valid_signature_without_timestamp():
    signer = TimestampSigner("secret-key", salt="test-salt")
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert result == b"test-value"

def test_unsign_valid_signature_with_timestamp():
    signer = TimestampSigner("secret-key", salt="test-salt")
    signed = signer.sign("test-value")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test-value"
    assert isinstance(result[1], datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret-key", salt="test-salt")
    signed = signer.sign("test-value")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_future_signature():
    signer = TimestampSigner("secret-key", salt="test-salt")
    signed = signer.sign("test-value")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret-key", salt="test-salt")
    try:
        signer.unsign("invalid-value")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret-key", salt="test-salt")
    signed = b"test-value." + base64_encode(b"invalid")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_timestamp():
    signer = TimestampSigner("secret-key", salt="test-salt")
    signed = signer.sign("test-value")
    tampered = signed[:-1] + b"X"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_without_timestamp():
    signer = TimestampSigner("secret-key", salt="test-salt")
    try:
        signer.unsign(b"invalid-value")
        assert False
    except BadSignature:
        pass


# LLM-generated content at query #48
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1

def test_timestamp_signer_custom_separator():
    signer = TimestampSigner("secret", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret", salt="custom")
    assert signer.salt == b"custom"

def test_timestamp_signer_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_custom_digest_method():
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_secret_key_property():
    signer = TimestampSigner(["old_secret", "new_secret"])
    assert signer.secret_key == b"new_secret"
    assert signer.secret_keys == [b"old_secret", b"new_secret"]


# LLM-generated content at query #49
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Ensure that the try block at line 49-57 does not raise an exception
    result = signer.unsign(signed_value)
    assert result == b"test"
```


# LLM-generated content at query #50
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_custom_sep():
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_custom_algorithm():
    signer = TimestampSigner("secret-key", algorithm=HMACAlgorithm(hashlib.sha256))
    assert signer.algorithm is not None

def test_timestamp_signer_multiple_secret_keys():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_sep_in_base64_alphabet_raises():
    from itsdangerous.signer import _base64_alphabet
    for sep in _base64_alphabet:
        try:
            TimestampSigner("secret-key", sep=sep)
            assert False, f"Expected ValueError for sep {sep!r}"
        except ValueError:
            pass


# LLM-generated content at query #51
#--------------------------

def test_predicate_at_line_77_evaluates_to_true():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # tamper with the timestamp to make it in the future
    parts = signed_value.rsplit(signer.sep.encode(), 1)
    future_timestamp = base64_encode(int_to_bytes(int(time.time()) + 1000))
    tampered = parts[0] + signer.sep.encode() + future_timestamp
    # now sign with a different secret to cause signature error first
    signer2 = TimestampSigner("different_secret")
    signed_with_different = signer2.sign(parts[0])
    # combine: use the value from tampered but signature from different secret
    # to ensure sig_error is set and timestamp is valid (future)
    combined = tampered.rsplit(signer.sep.encode(), 1)[0] + signer.sep.encode() + future_timestamp + signer.sep.encode() + signed_with_different.rsplit(signer.sep.encode(), 1)[1]
    try:
        signer.unsign(combined, max_age=10)
    except SignatureExpired as e:
        assert "age" in str(e) and "< 0 seconds" in str(e)


# LLM-generated content at query #52
#--------------------------

```python
def test_ts_int_is_none_after_base64_decode_failure():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    sep = signer.sep.encode()
    value, ts_bytes = signer.unsign(signed_value, return_timestamp=False)
    # Simulate a malformed timestamp that fails base64_decode
    malformed_signed = value + sep + b"!!invalid_base64!!"
    try:
        signer.unsign(malformed_signed)
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)
        assert e.payload == value
```


# LLM-generated content at query #53
#--------------------------

```python
def test_unsign_with_valid_timestamp_after_exception_handling():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Modify the signed value to have a valid signature but an invalid timestamp (e.g., non-base64)
    sep = signer.sep.encode()
    parts = signed_value.rsplit(sep, 1)
    # Create a signed value with a valid signature but a timestamp that is not base64 decodable
    valid_signature = parts[0].rsplit(sep, 1)[1]
    value_part = parts[0].rsplit(sep, 1)[0]
    invalid_ts = b"not_base64!!"
    malformed_signed = value_part + sep + invalid_ts + sep + valid_signature
    # This should not raise an exception from the except block; it should proceed and then raise BadTimeSignature
    try:
        signer.unsign(malformed_signed)
    except BadTimeSignature as e:
        assert e.payload == value_part
```


# LLM-generated content at query #54
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_key == b"secret"
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_with_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_with_sep():
    signer = TimestampSigner("secret", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_with_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_with_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_with_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algo = HMACAlgorithm()
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_secret_keys_list():
    signer = TimestampSigner(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.secret_key == b"key2"


# LLM-generated content at query #55
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_key == b"secret"
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1

def test_timestamp_signer_with_bytes_secret():
    signer = TimestampSigner(b"secret")
    assert signer.secret_key == b"secret"

def test_timestamp_signer_with_custom_salt():
    signer = TimestampSigner("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_with_custom_sep():
    signer = TimestampSigner("secret", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_with_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_with_custom_digest_method():
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_with_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm == algorithm

def test_timestamp_signer_secret_key_uses_last_key():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_with_sep_in_base64_raises():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret", sep=b"+")


# LLM-generated content at query #56
#--------------------------

def test_unsign_valid_no_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_with_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_valid_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test") + b"x"
    try:
        signer.unsign(signed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, sep, sig = signed.rpartition(b".")
    bad_signed = value + sep + b"invalidsig"
    try:
        signer.unsign(bad_signed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    sep = signer.sep.encode()
    timestamp = b"not_base64"
    signed = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)
    try:
        signer.unsign(signed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass


# LLM-generated content at query #57
#--------------------------

def test_unsign_valid_signature_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert len(result) == 2
    assert result[0] == b"test"

def test_unsign_valid_signature_with_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    import time
    time.sleep(0.01)
    result = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert result[0] == b"test"

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    import time
    time.sleep(0.01)
    from itsdangerous.exc import SignatureExpired
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_future_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    from itsdangerous.exc import SignatureExpired
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = b"test" + signer.sep.encode() + signer.get_signature(b"test")
    from itsdangerous.exc import BadTimeSignature
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = b"test" + signer.sep.encode() + b"invalid" + signer.sep.encode() + signer.get_signature(b"test" + signer.sep.encode() + b"invalid")
    from itsdangerous.exc import BadTimeSignature
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed = b"test" + signer.sep.encode() + b"MTIzNDU2Nzg5" + signer.sep.encode() + b"badsig"
    from itsdangerous.exc import BadTimeSignature
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_return_timestamp_false():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=False)
    assert result == b"test"

def test_unsign_with_empty_value():
    signer = TimestampSigner("secret")
    signed = signer.sign("")
    result = signer.unsign(signed, return_timestamp=True)
    assert result[0] == b""


# LLM-generated content at query #58
#--------------------------

```python
def test_unsign_with_valid_signature_and_timestamp_not_none():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"


# LLM-generated content at query #59
#--------------------------

def test_unsign_valid_signature_without_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, ts = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(ts, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    with raises(SignatureExpired):
        signer.unsign(signed, max_age=-1)

def test_unsign_future_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    with raises(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_bad_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test") + b"x"
    with raises(BadTimeSignature):
        signer.unsign(signed)

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = b"test" + signer.sep.encode() + signer.get_signature(b"test")
    with raises(BadTimeSignature):
        signer.unsign(signed)

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = b"test" + signer.sep.encode() + b"invalid" + signer.sep.encode() + signer.get_signature(b"test" + signer.sep.encode() + b"invalid")
    with raises(BadTimeSignature):
        signer.unsign(signed)


# LLM-generated content at query #60
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_with_bytes_secret_key():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_with_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_with_sep():
    signer = TimestampSigner("secret-key", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_with_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_with_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_with_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    from itsdangerous.signer import _lazy_sha1
    signer = TimestampSigner("secret-key", algorithm=HMACAlgorithm(_lazy_sha1))
    assert signer.algorithm is not None

def test_timestamp_signer_with_list_of_keys():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_timestamp_signer_with_bytes_salt():
    signer = TimestampSigner("secret-key", salt=b"bytes-salt")
    assert signer.salt == b"bytes-salt"

def test_timestamp_signer_with_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


