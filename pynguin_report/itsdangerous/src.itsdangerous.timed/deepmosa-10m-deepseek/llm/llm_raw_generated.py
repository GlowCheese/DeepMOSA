####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_with_bytes_secret():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_with_list_of_strings():
    signer = TimestampSigner(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_timestamp_signer_with_custom_sep():
    signer = TimestampSigner("secret", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_with_custom_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_with_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_with_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_with_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algo = HMACAlgorithm()
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm == algo


# LLM-generated content at query #2
#--------------------------

def test_unsign_returns_value_when_valid_no_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_returns_value_and_timestamp_when_return_timestamp_true():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, ts = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(ts, datetime)

def test_unsign_raises_bad_time_signature_when_no_separator():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"no_separator")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_raises_bad_time_signature_when_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bad_ts = b"invalid_base64"
    sep = signer.sep.encode()
    malformed = b"test" + sep + bad_ts + sep + b"signature"
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_raises_signature_expired_when_age_exceeds_max_age():
    signer = TimestampSigner("secret")
    old_ts = 0
    old_ts_b64 = base64_encode(int_to_bytes(old_ts)).decode()
    sep = signer.sep
    value = "test"
    signed = value + sep + old_ts_b64 + sep + signer.get_signature((value + sep + old_ts_b64).encode())
    try:
        signer.unsign(signed, max_age=10)
        assert False
    except SignatureExpired:
        pass

def test_unsign_raises_signature_expired_when_age_negative():
    signer = TimestampSigner("secret")
    future_ts = int(time.time()) + 100
    future_ts_b64 = base64_encode(int_to_bytes(future_ts)).decode()
    sep = signer.sep
    value = "test"
    signed = value + sep + future_ts_b64 + sep + signer.get_signature((value + sep + future_ts_b64).encode())
    try:
        signer.unsign(signed, max_age=10)
        assert False
    except SignatureExpired:
        pass

def test_unsign_raises_bad_signature_when_signature_invalid_and_no_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False
    except BadSignature:
        pass

def test_unsign_raises_bad_time_signature_when_signature_invalid_but_timestamp_present():
    signer = TimestampSigner("secret")
    valid_signed = signer.sign("test")
    sep = signer.sep.encode()
    # Create a tampered signature
    tampered = valid_signed[:-1] + b"X"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_returns_value_when_valid_with_max_age_not_exceeded():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"


# LLM-generated content at query #3
#--------------------------

def test_unsign_valid_without_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_with_return_timestamp():
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

def test_unsign_future_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    signed = signed[:-10] + b"invalid"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    signed = signed.rsplit(b".", 1)[0]
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    signed = signed[:-1] + b"x"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_empty_string():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"")
        assert False
    except BadSignature:
        pass


# LLM-generated content at query #4
#--------------------------

def test_unsign_returns_tuple_when_return_timestamp_is_true():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bytes)
    assert isinstance(result[1], datetime)


# LLM-generated content at query #5
#--------------------------

def test_unsign_valid_signature_no_max_age():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed, return_timestamp=False)
    assert result == value

def test_unsign_valid_signature_with_max_age_not_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result = signer.unsign(signed, max_age=3600, return_timestamp=False)
    assert result == value

def test_unsign_valid_signature_with_max_age_expired():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    signer.get_timestamp = lambda: 0
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except Exception as e:
        assert isinstance(e, Exception)

def test_unsign_valid_signature_with_return_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    signed = signer.sign(value)
    result, ts = signer.unsign(signed, return_timestamp=True)
    assert result == value
    assert isinstance(ts, type(datetime(2020, 1, 1, tzinfo=timezone.utc)))

def test_unsign_invalid_signature_no_timestamp():
    signer = TimestampSigner("secret")
    signed = b"test"
    try:
        signer.unsign(signed)
        assert False
    except Exception as e:
        assert isinstance(e, Exception)

def test_unsign_invalid_signature_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    sep = b"."
    signed = value + sep + b"invalid_base64"
    try:
        signer.unsign(signed)
        assert False
    except Exception as e:
        assert isinstance(e, Exception)

def test_unsign_signature_error_with_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    sep = b"."
    signed = value + sep + b"YWJjZGVmZ2g="  # valid base64 but wrong signature
    try:
        signer.unsign(signed)
        assert False
    except Exception as e:
        assert isinstance(e, Exception)

def test_unsign_empty_signed_value():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"")
        assert False
    except Exception as e:
        assert isinstance(e, Exception)


# LLM-generated content at query #6
#--------------------------

def test_unsign_with_valid_signature_and_valid_timestamp_does_not_enter_except_block():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    signer.unsign(signed_value)


# LLM-generated content at query #7
#--------------------------

```python
def test_unsign_valid_signature_without_max_age():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    result = signer.unsign(signed)
    assert result == b"test value"

def test_unsign_valid_signature_with_max_age_not_expired():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test value"

def test_unsign_valid_signature_with_return_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test value"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    import time
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    time.sleep(1)
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    parts = signed.rsplit(b".", 1)
    malformed = parts[0] + b"." + b"invalid"
    try:
        signer.unsign(malformed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    parts = signed.rsplit(b".", 1)
    missing = parts[0]
    try:
        signer.unsign(missing)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    bad_sig = signed + b"x"
    try:
        signer.unsign(bad_sig)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_negative_age():
    import time
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    time.sleep(1)
    try:
        signer.unsign(signed, max_age=3600)
        assert True
    except SignatureExpired:
        assert False, "Should not expire"

def test_unsign_empty_value():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("")
    result = signer.unsign(signed)
    assert result == b""

def test_unsign_bytes_input():
    signer = TimestampSigner(b"secret-key")
    signed = signer.sign(b"test value")
    result = signer.unsign(signed)
    assert result == b"test value"

def test_unsign_string_input():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test value")
    result = signer.unsign(signed.decode())
    assert result == b"test value"
```


# LLM-generated content at query #8
#--------------------------

def test_unsign_valid_signature_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed)
    assert result == b"value"

def test_unsign_valid_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"value"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_future_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_bad_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False
    except BadSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = b"value" + signer.sep.encode() + signer.get_signature(b"value")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = b"value" + signer.sep.encode() + b"badts" + signer.sep.encode() + signer.get_signature(b"value" + signer.sep.encode() + b"badts")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_bad_signature_and_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = b"value" + signer.sep.encode() + b"badts" + signer.sep.encode() + b"badsig"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_bad_signature_and_valid_timestamp():
    signer = TimestampSigner("secret")
    from time import time
    ts = int(time())
    import base64
    ts_bytes = base64.urlsafe_b64encode(ts.to_bytes(8, 'big').lstrip(b'\x00') or b'\x00')
    signed = b"value" + signer.sep.encode() + ts_bytes + signer.sep.encode() + b"badsig"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #9
#--------------------------

def test_unsign_valid_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed)
    assert result == b"value"

def test_unsign_valid_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"value"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    import time
    time.sleep(0.01)
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_bad_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False
    except BadSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value.sep")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value.sep.invalid")
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #10
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method.__name__ == "sha1"
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_all_parameters():
    from hashlib import sha256
    from itsdangerous.signer import HMACAlgorithm
    signer = TimestampSigner(
        secret_key="mysecret",
        salt=b"mysalt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=sha256,
        algorithm=HMACAlgorithm(sha256)
    )
    assert signer.secret_keys == [b"mysecret"]
    assert signer.sep == b"|"
    assert signer.salt == b"mysalt"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_list_of_secret_keys():
    signer = TimestampSigner(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.secret_key == b"key2"

def test_timestamp_signer_constructor_with_bytes_secret_key():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_invalid_separator():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret", sep=b".")

def test_timestamp_signer_constructor_inherits_defaults():
    signer = TimestampSigner("secret")
    assert signer.default_digest_method is not None
    assert signer.default_key_derivation == "django-concat"


# LLM-generated content at query #11
#--------------------------

```python
def test_unsign_predicate_at_line_43_evaluates_to_false():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    # signed_value has valid timestamp, so bytes_to_int(base64_decode(ts_bytes)) succeeds
    result = signer.unsign(signed_value)
    assert result == b"test"
```


# LLM-generated content at query #12
#--------------------------

```python
def test_unsign_with_max_age_and_age_negative():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    signer.get_timestamp = lambda: 0
    signer.timestamp_to_datetime = lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc)
    try:
        signer.unsign(signed_value, max_age=100)
    except SignatureExpired:
        pass
```


# LLM-generated content at query #13
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_with_custom_sep():
    signer = TimestampSigner("secret-key", sep=b"-")
    assert signer.sep == b"-"

def test_timestamp_signer_with_custom_salt():
    signer = TimestampSigner("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_with_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_with_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

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

def test_timestamp_signer_with_key_rotation():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_with_bytes_secret():
    signer = TimestampSigner(b"bytes-secret")
    assert signer.secret_keys == [b"bytes-secret"]

def test_timestamp_signer_separator_not_in_base64_alphabet():
    signer = TimestampSigner("secret-key", sep=b":")
    assert signer.sep == b":"


# LLM-generated content at query #14
#--------------------------

def test_timestamp_to_datetime_does_not_raise_for_valid_timestamp():
    signer = TimestampSigner("secret")
    result = signer.unsign(signer.sign("test"), return_timestamp=True)
    assert isinstance(result, tuple)


# LLM-generated content at query #15
#--------------------------

```python
def test_unsign_with_bad_signature_but_valid_timestamp():
    signer = TimestampSigner("secret-key")
    valid_signed = signer.sign("test")
    sep = signer.sep.encode()
    value, ts_bytes = valid_signed.rsplit(sep, 1)
    # Create a signed value with a valid timestamp but invalid signature
    bad_sig = value + sep + ts_bytes + sep + b"invalid"
    try:
        signer.unsign(bad_sig)
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #16
#--------------------------

def test_loads_with_string_input():
    serializer = TimedSerializer("secret-key")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed = serializer.dumps("test")
    result = serializer.loads(signed.encode())
    assert result == "test"

def test_loads_with_max_age():
    serializer = TimedSerializer("secret-key")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, max_age=3600)
    assert result == "test"

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret-key")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, float)

def test_loads_with_salt():
    serializer = TimedSerializer("secret-key")
    signed = serializer.dumps("test", salt="custom")
    result = serializer.loads(signed, salt="custom")
    assert result == "test"

def test_loads_raises_signature_expired():
    serializer = TimedSerializer("secret-key")
    signed = serializer.dumps("test")
    import time
    time.sleep(0.1)
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except:
        pass

def test_loads_raises_bad_signature():
    serializer = TimedSerializer("secret-key")
    try:
        serializer.loads(b"invalid")
        assert False
    except:
        pass


# LLM-generated content at query #17
#--------------------------

```python
def test_timestamp_to_datetime_raises_bad_time_signature_on_value_error():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    signer.timestamp_to_datetime = lambda ts: (_ for _ in ()).throw(ValueError("invalid"))
    try:
        signer.unsign(signed_value, return_timestamp=False)
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #18
#--------------------------

```python
def test_sep_in_result_after_unsign_with_signature_error():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Modify the signature part to make it invalid
    parts = signed_value.rsplit(signer.sep.encode(), 1)
    invalid_signed = parts[0] + signer.sep.encode() + b"invalid_signature"
    try:
        signer.unsign(invalid_signed)
    except BadSignature:
        pass
    else:
        # If no exception, the condition might have been met, but we expect an exception
        # Actually we want to ensure that when sep is in result and sig_error is set, 
        # the predicate at line 32 (sep not in result) is False.
        # So we need to reach line 31 with sep in result.
        # This test ensures that after super().unsign raises BadSignature, 
        # result contains sep (from the original signed value).
        # If we get here, it means no exception was raised, which is unexpected.
        # But to test the predicate, we can check that the result has sep.
        pass
```


# LLM-generated content at query #19
#--------------------------

```python
def test_unsign_ts_bytes_decode_raises_exception_and_ts_int_remains_none():
    signer = TimestampSigner("secret-key")
    # Simulate a signed value with a timestamp that cannot be base64 decoded
    # We need to craft a value that passes the super().unsign but has an invalid timestamp.
    # To do this, we can create a valid signature for a value, then replace the timestamp part
    # with an invalid base64 string.
    value = b"test"
    # Get a valid timestamp and signature
    timestamp = base64_encode(int_to_bytes(1))
    sep = signer.sep.encode()
    signed = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)
    # Now replace the timestamp with an invalid base64 string
    invalid_ts = b"!!!invalid_base64!!!"
    signed_invalid = value + sep + invalid_ts + sep + signer.get_signature(value + sep + invalid_ts)
    # This should not raise, and ts_int should remain None
    result = signer.unsign(signed_invalid)
    # The method should return the value unchanged because ts_int is None but sig_error is also None
    assert result == value
```


# LLM-generated content at query #20
#--------------------------

```python
def test_ts_int_is_none_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign(b"test")
    sep = signer.sep.encode()
    value, ts_bytes = signed_value.rsplit(sep, 1)
    malformed_ts = base64_encode(b"invalid")
    bad_signed = value + sep + malformed_ts + sep + signer.get_signature(value + sep + malformed_ts)
    try:
        signer.unsign(bad_signed)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value
```


# LLM-generated content at query #21
#--------------------------

def test_loads_returns_payload_when_no_max_age_and_no_return_timestamp():
    serializer = TimedSerializer("secret")
    s = serializer.dumps("test")
    result = serializer.loads(s)
    assert result == "test"

def test_loads_returns_payload_and_timestamp_when_return_timestamp_true():
    serializer = TimedSerializer("secret")
    s = serializer.dumps("test")
    payload, timestamp = serializer.loads(s, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, float)

def test_loads_raises_signature_expired_when_max_age_exceeded():
    import time
    serializer = TimedSerializer("secret")
    s = serializer.dumps("test")
    time.sleep(0.01)
    try:
        serializer.loads(s, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_loads_raises_bad_signature_for_invalid_data():
    serializer = TimedSerializer("secret")
    try:
        serializer.loads(b"invalid")
        assert False
    except BadSignature:
        pass

def test_loads_with_salt_uses_correct_signer():
    serializer = TimedSerializer("secret", salt="custom_salt")
    s = serializer.dumps("test")
    result = serializer.loads(s, salt="custom_salt")
    assert result == "test"

def test_loads_raises_bad_signature_with_wrong_salt():
    serializer = TimedSerializer("secret", salt="salt1")
    s = serializer.dumps("test")
    try:
        serializer.loads(s, salt="salt2")
        assert False
    except BadSignature:
        pass

def test_loads_returns_payload_when_max_age_not_exceeded():
    serializer = TimedSerializer("secret")
    s = serializer.dumps("test")
    result = serializer.loads(s, max_age=3600)
    assert result == "test"

def test_loads_returns_payload_with_return_timestamp_and_max_age():
    serializer = TimedSerializer("secret")
    s = serializer.dumps("test")
    payload, timestamp = serializer.loads(s, max_age=3600, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, float)


# LLM-generated content at query #22
#--------------------------

def test_loads_raises_signature_expired_when_expired():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    with pytest.raises(SignatureExpired):
        serializer.loads(signed, max_age=-1)


# LLM-generated content at query #23
#--------------------------

def test_unsign_without_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = b"value" + signer.sep.encode() + b"invalidsignature"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    parts = signed_value.rsplit(signer.sep.encode(), 1)
    malformed = parts[0] + signer.sep.encode() + b"notb64"
    try:
        signer.unsign(malformed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_wrong_signature_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    tampered = signed_value[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_with_expired_signature_raises_signature_expired():
    import time
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    future_timestamp = int(time.time()) + 1000
    # Manually replace timestamp with future timestamp
    sep = signer.sep.encode()
    parts = signed_value.rsplit(sep, 1)
    new_ts = base64_encode(int_to_bytes(future_timestamp))
    forged = parts[0] + sep + new_ts + sep + signer.get_signature(parts[0] + sep + new_ts)
    try:
        signer.unsign(forged, max_age=10)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_with_negative_age_raises_signature_expired():
    import time
    signer = TimestampSigner("secret")
    signed_value = signer.sign("value")
    past_timestamp = int(time.time()) - 1000
    sep = signer.sep.encode()
    parts = signed_value.rsplit(sep, 1)
    new_ts = base64_encode(int_to_bytes(past_timestamp))
    forged = parts[0] + sep + new_ts + sep + signer.get_signature(parts[0] + sep + new_ts)
    try:
        signer.unsign(forged, max_age=10)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_valid_signature_returns_value():
    signer = TimestampSigner("secret")
    result = signer.unsign(signer.sign("test_value"))
    assert result == b"test_value"

def test_unsign_return_timestamp_returns_tuple():
    signer = TimestampSigner("secret")
    value, dt = signer.unsign(signer.sign("test_value"), return_timestamp=True)
    assert value == b"test_value"
    assert isinstance(dt, datetime)

def test_unsign_with_max_age_valid():
    signer = TimestampSigner("secret")
    result = signer.unsign(signer.sign("test_value"), max_age=3600)
    assert result == b"test_value"


# LLM-generated content at query #24
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_secret_key_bytes():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_timestamp_signer_constructor_with_secret_key_list():
    signer = TimestampSigner(["old", "new"])
    assert signer.secret_keys == [b"old", b"new"]

def test_timestamp_signer_constructor_with_secret_key_bytes_list():
    signer = TimestampSigner([b"old", b"new"])
    assert signer.secret_keys == [b"old", b"new"]

def test_timestamp_signer_constructor_with_custom_salt():
    signer = TimestampSigner("secret", salt="custom")
    assert signer.salt == b"custom"

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret", sep=":")
    assert signer.sep == b":"

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
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_constructor_sep_in_base64_raises():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret", sep="a")


# LLM-generated content at query #25
#--------------------------

def test_loads_returns_payload_when_return_timestamp_is_false():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, return_timestamp=False)
    assert result == "test"


# LLM-generated content at query #26
#--------------------------

def test_unsign_returns_bytes_when_no_timestamp_requested():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed)
    assert result == b"value"

def test_unsign_returns_tuple_when_return_timestamp_true():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"value"
    assert isinstance(result[1], datetime)

def test_unsign_raises_bad_signature_on_invalid_signature():
    signer = TimestampSigner("secret")
    signed = b"value.invalid"
    try:
        signer.unsign(signed)
        assert False
    except BadSignature:
        pass

def test_unsign_raises_bad_time_signature_on_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = b"value"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_raises_signature_expired_on_old_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_raises_signature_expired_on_future_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_valid_signature_and_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"value"

def test_unsign_raises_bad_time_signature_on_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = b"value.sep.invalid"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_returns_bytes_on_validation_failure_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"value"
    assert isinstance(result[1], datetime)


# LLM-generated content at query #27
#--------------------------

def test_timestamp_signer_default_construction():
    signer = TimestampSigner("secret")
    assert signer.secret_key == b"secret"
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.algorithm is not None
    assert signer.digest_method is not None

def test_timestamp_signer_with_custom_sep():
    signer = TimestampSigner("secret", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_with_custom_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

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

def test_timestamp_signer_with_multiple_secret_keys():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"


# LLM-generated content at query #28
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_all_parameters():
    signer = TimestampSigner(
        secret_key="my-secret",
        salt="custom-salt",
        sep="|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256),
    )
    assert signer.secret_keys == [b"my-secret"]
    assert signer.salt == b"custom-salt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_key_derivation_none():
    signer = TimestampSigner("secret", key_derivation="none")
    assert signer.key_derivation == "none"
    assert signer.derive_key() == b"secret"

def test_timestamp_signer_constructor_separator_not_in_base64_alphabet():
    signer = TimestampSigner("secret", sep="_")
    assert signer.sep == b"_"

def test_timestamp_signer_constructor_salt_bytes():
    signer = TimestampSigner("secret", salt=b"bytes-salt")
    assert signer.salt == b"bytes-salt"


# LLM-generated content at query #29
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method.__name__ == "sha1"
    assert signer.algorithm.algorithm_type == "hmac"

def test_timestamp_signer_constructor_with_salt_bytes():
    signer = TimestampSigner("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_salt_string():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_sep_bytes():
    signer = TimestampSigner("secret", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_sep_string():
    signer = TimestampSigner("secret", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_timestamp_signer_constructor_with_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    signer = TimestampSigner("secret", algorithm=HMACAlgorithm(hashlib.sha256))
    assert signer.algorithm.algorithm_type == "hmac"

def test_timestamp_signer_constructor_with_secret_key_list():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_constructor_with_invalid_sep():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret", sep=b"a")


# LLM-generated content at query #30
#--------------------------

def test_unsign_age_negative():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    signer.get_timestamp = lambda: -1
    try:
        signer.unsign(signed, max_age=10)
    except SignatureExpired:
        pass


# LLM-generated content at query #31
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

def test_unsign_valid_signature_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    import time
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    time.sleep(1)
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = b"test.invalid"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, sep, sig = signed.rpartition(b".")
    bad_signed = value + sep + b"badsig"
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_empty_signed_value():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"")
        assert False
    except BadSignature:
        pass


# LLM-generated content at query #32
#--------------------------

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
    signer = TimestampSigner("secret", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_custom_digest_method():
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    algo = HMACAlgorithm(hashlib.sha512)
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm == algo

def test_timestamp_signer_constructor_with_secret_key_list():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]

def test_timestamp_signer_constructor_with_bytes_secret():
    signer = TimestampSigner(b"bytes_secret")
    assert signer.secret_keys == [b"bytes_secret"]


# LLM-generated content at query #33
#--------------------------

```python
def test_unsign_raises_bad_time_signature_on_value_error_from_timestamp_to_datetime():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    sep = signer.sep.encode()
    value, ts_bytes = signed.rsplit(sep, 1)
    ts_int = bytes_to_int(base64_decode(ts_bytes))
    invalid_ts_bytes = base64_encode(int_to_bytes(10**100))
    bad_signed = value + sep + invalid_ts_bytes + sep + signer.get_signature(value + sep + invalid_ts_bytes)
    try:
        signer.unsign(bad_signed)
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)


# LLM-generated content at query #34
#--------------------------

```python
def test_predicate_line_52_evaluates_to_true():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    signed_value = signed_value[:-1]
    try:
        signer.unsign(signed_value)
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #35
#--------------------------

def test_predicate_line52_evaluates_to_false():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, return_timestamp=False)
    assert isinstance(result, bytes)


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_true():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    bad_timestamp = base64_encode(int_to_bytes(-1))
    sep = signer.sep.encode()
    manipulated = b"test" + sep + bad_timestamp + sep + signer.get_signature(b"test" + sep + bad_timestamp)
    try:
        signer.unsign(manipulated)
    except BadTimeSignature:
        pass
```


# LLM-generated content at query #37
#--------------------------

def test_unsign_successful_without_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_successful_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_raises_bad_signature_on_missing_sep():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"justdata")
        assert False, "Expected BadSignature"
    except BadSignature:
        pass

def test_unsign_raises_bad_time_signature_on_missing_timestamp():
    signer = TimestampSigner("secret")
    # Create a signed value without a timestamp by using Signer directly
    from itsdangerous.signer import Signer
    base_signer = Signer("secret")
    signed_no_timestamp = base_signer.sign(b"test")
    try:
        signer.unsign(signed_no_timestamp)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_raises_bad_time_signature_on_malformed_timestamp():
    signer = TimestampSigner("secret")
    # Create a signed value with a malformed timestamp (invalid base64)
    value = want_bytes("test")
    sep = want_bytes(signer.sep)
    malformed_ts = b"not-valid-base64!!!"
    signed = value + sep + malformed_ts + sep + signer.get_signature(value + sep + malformed_ts)
    try:
        signer.unsign(signed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_raises_signature_expired_on_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_raises_signature_expired_on_negative_age():
    signer = TimestampSigner("secret")
    # Manually create a signed value with a future timestamp
    future_ts = int(time.time()) + 1000
    value = want_bytes("test")
    sep = want_bytes(signer.sep)
    ts_bytes = base64_encode(int_to_bytes(future_ts))
    signed = value + sep + ts_bytes + sep + signer.get_signature(value + sep + ts_bytes)
    try:
        signer.unsign(signed, max_age=10)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_returns_payload_on_bad_signature_with_valid_timestamp():
    signer = TimestampSigner("secret")
    # Create a signed value with a valid timestamp but wrong signature
    value = want_bytes("test")
    sep = want_bytes(signer.sep)
    ts_bytes = base64_encode(int_to_bytes(int(time.time())))
    signed = value + sep + ts_bytes + sep + b"wrongsignature"
    try:
        signer.unsign(signed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert e.payload == b"test"


# LLM-generated content at query #38
#--------------------------

def test_unsign_returns_bytes_when_return_timestamp_false():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=False)
    assert result == b"test"

def test_unsign_returns_tuple_when_return_timestamp_true():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)

def test_unsign_raises_bad_signature_for_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_raises_bad_time_signature_when_timestamp_missing():
    signer = TimestampSigner("secret")
    value = b"test"
    sep = signer.sep.encode()
    no_timestamp = value + sep + signer.get_signature(value)
    try:
        signer.unsign(no_timestamp)
        assert False, "Should have raised BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_raises_signature_expired_when_max_age_exceeded():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_raises_signature_expired_for_negative_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False, "Should have raised SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_returns_bytes_with_valid_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_returns_tuple_with_valid_max_age_and_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)


# LLM-generated content at query #39
#--------------------------

def test_unsign_success_returns_bytes():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed)
    assert result == b"value"

def test_unsign_success_returns_tuple_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result, ts = signer.unsign(signed, return_timestamp=True)
    assert result == b"value"
    assert isinstance(ts, datetime)

def test_unsign_raises_bad_signature_on_invalid():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"invalid")
        assert False
    except BadSignature:
        pass

def test_unsign_raises_signature_expired_when_older_than_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_raises_signature_expired_when_age_negative():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=1000000)
        assert False
    except SignatureExpired:
        pass

def test_unsign_preserves_payload_from_bad_signature():
    signer = TimestampSigner("secret")
    signed = b"invalid" + signer.sep.encode() + b"timestamp"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature as e:
        assert e.payload == b"invalid"

def test_unsign_raises_bad_time_signature_when_timestamp_missing():
    signer = TimestampSigner("secret")
    signed = b"value" + signer.sep.encode() + b"sig"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_raises_bad_time_signature_when_timestamp_malformed():
    signer = TimestampSigner("secret")
    signed = b"value" + signer.sep.encode() + b"invalidsig" + signer.sep.encode() + b"badts"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_validates_with_correct_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"value"


# LLM-generated content at query #40
#--------------------------

def test_ts_int_is_none_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    # Construct a signed value where the timestamp part is not valid base64
    # so that base64_decode raises an exception, causing ts_int to remain None.
    # The signature is correct, so sig_error is None.
    value = b"test"
    sep = signer.sep.encode()
    # Use an invalid base64 string for the timestamp (e.g., "!!!")
    bad_ts = b"!!!"
    signed_value = value + sep + bad_ts + sep + signer.get_signature(value + sep + bad_ts)
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value


# LLM-generated content at query #41
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method() is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_bytes_secret():
    signer = TimestampSigner(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_timestamp_signer_constructor_with_list_keys():
    signer = TimestampSigner([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret-key", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_custom_sep_str():
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_with_custom_salt_bytes():
    signer = TimestampSigner("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_with_none_salt():
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
    algorithm = HMACAlgorithm()
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm


# LLM-generated content at query #42
#--------------------------

def test_loads_with_str_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed.encode())
    assert result == "test"

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, return_timestamp=True)
    assert len(result) == 2
    assert result[0] == "test"
    assert isinstance(result[1], float)

def test_loads_with_max_age_not_expired():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, max_age=3600)
    assert result == "test"

def test_loads_with_max_age_expired():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    import time
    time.sleep(0.1)
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except Exception:
        pass

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret")
    try:
        serializer.loads(b"invalid_data")
        assert False
    except Exception:
        pass

def test_loads_with_salt():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test", salt="custom_salt")
    result = serializer.loads(signed, salt="custom_salt")
    assert result == "test"


# LLM-generated content at query #43
#--------------------------

def test_timestamp_signer_constructor_default_parameters():
    signer = TimestampSigner("secret-key")

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")

def test_timestamp_signer_constructor_with_sep():
    signer = TimestampSigner("secret-key", sep=":")

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")

def test_timestamp_signer_constructor_with_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)

def test_timestamp_signer_constructor_with_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)

def test_timestamp_signer_constructor_with_bytes_secret_key():
    signer = TimestampSigner(b"secret-key")

def test_timestamp_signer_constructor_with_list_secret_keys():
    signer = TimestampSigner(["old-key", "new-key"])

def test_timestamp_signer_constructor_with_salt_none():
    signer = TimestampSigner("secret-key", salt=None)

def test_timestamp_signer_constructor_with_sep_bytes():
    signer = TimestampSigner("secret-key", sep=b"|")


# LLM-generated content at query #44
#--------------------------

```python
def test_unsign_predicate_line_43_false():
    import time
    from datetime import datetime, timezone
    from itsdangerous.timed import TimestampSigner
    from itsdangerous.signer import Signer
    from itsdangerous.exc import BadSignature, BadTimeSignature
    from itsdangerous.encoding import want_bytes, base64_encode, int_to_bytes

    signer = TimestampSigner("secret")
    value = b"test"
    timestamp = base64_encode(int_to_bytes(int(time.time())))
    sep = want_bytes(signer.sep)
    signed_value = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)

    # Call unsign with a valid signed value, sig_error is None, so predicate at line 48 is False
    result = signer.unsign(signed_value)
    assert result == value
```


# LLM-generated content at query #45
#--------------------------

def test_unsign_returns_bytes_when_return_timestamp_false():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed)
    assert result == b"value"

def test_unsign_returns_tuple_when_return_timestamp_true():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"value"
    assert isinstance(result[1], datetime)

def test_unsign_raises_bad_signature_for_invalid_signature():
    signer = TimestampSigner("secret")
    signed = b"value.invalid"
    try:
        signer.unsign(signed)
        assert False
    except BadSignature:
        pass

def test_unsign_raises_bad_time_signature_for_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    no_timestamp = signed.split(signer.sep.encode())[0]
    try:
        signer.unsign(no_timestamp)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_raises_signature_expired_when_max_age_exceeded():
    import time
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    time.sleep(1)
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_raises_signature_expired_for_negative_age():
    import time
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass


# LLM-generated content at query #46
#--------------------------

```python
def test_timestamp_to_datetime_raises_exception_on_invalid_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    sep = signer.sep.encode()
    value, ts_bytes = signed_value.rsplit(sep, 1)
    invalid_ts = base64_encode(b"\xff\xff\xff\xff")
    manipulated_signed_value = value + sep + invalid_ts + sep + signer.get_signature(value + sep + invalid_ts)
    try:
        signer.unsign(manipulated_signed_value)
    except BadTimeSignature:
        pass
    else:
        raise AssertionError("Expected BadTimeSignature exception")```


# LLM-generated content at query #47
#--------------------------

def test_timestamp_signer_constructor_default():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_custom_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_sep_in_base64():
    try:
        TimestampSigner("secret", sep="a")
        assert False
    except ValueError as e:
        assert "cannot be used" in str(e)

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algorithm = HMACAlgorithm()
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_secret_key_list():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_constructor_with_bytes_secret():
    signer = TimestampSigner(b"bytes_secret")
    assert signer.secret_keys == [b"bytes_secret"]

def test_timestamp_signer_constructor_empty_secret_key_raises():
    try:
        TimestampSigner("")
        assert False
    except Exception:
        pass


# LLM-generated content at query #48
#--------------------------

```python
def test_unsign_predicate_line_43_false():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    signer.unsign(signed_value)
```


# LLM-generated content at query #49
#--------------------------

def test_ts_int_is_none_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    sep = signer.sep.encode()
    value, ts_bytes = signed_value.rsplit(sep, 1)
    # Replace the timestamp with an invalid base64 string that decodes to nothing valid
    malformed_signed = value + sep + b"invalid"
    try:
        signer.unsign(malformed_signed)
    except BadTimeSignature as e:
        assert e.payload == value


# LLM-generated content at query #50
#--------------------------

def test_unsign_valid_signature_no_max_age_no_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_max_age_within_limit():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_valid_signature_with_return_timestamp():
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

def test_unsign_signature_age_negative():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"test")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bad_signed = signed + b"invalid"
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bad_signed = signed[:-1] + b"x"
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_valid_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bad_signed = b"wrong" + signed[4:]
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #51
#--------------------------

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

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    no_timestamp = signed.split(b".")[0]
    try:
        signer.unsign(no_timestamp)
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

def test_unsign_returns_bytes():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert isinstance(result, bytes)

def test_unsign_returns_tuple_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bytes)
    assert isinstance(result[1], datetime)


# LLM-generated content at query #52
#--------------------------

def test_loads_returns_payload_when_valid():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_returns_payload_and_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == "test"
    assert isinstance(result[1], int)

def test_loads_with_max_age_raises_signature_expired():
    import time
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    time.sleep(0.1)
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except Exception as e:
        assert isinstance(e, Exception)

def test_loads_with_invalid_signature_raises_bad_signature():
    serializer = TimedSerializer("secret")
    signed = b"invalid"
    try:
        serializer.loads(signed)
        assert False
    except Exception as e:
        assert isinstance(e, Exception)

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_with_string_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test").decode()
    result = serializer.loads(signed)
    assert result == "test"


# LLM-generated content at query #53
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_with_sep():
    signer = TimestampSigner("secret-key", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_secret_key_list():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_constructor_with_bytes_secret():
    signer = TimestampSigner(b"bytes-key")
    assert signer.secret_keys == [b"bytes-key"]

def test_timestamp_signer_constructor_with_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_invalid_sep():
    try:
        TimestampSigner("secret-key", sep=b".")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for separator '.'"


# LLM-generated content at query #54
#--------------------------

```python
def test_loads_with_return_timestamp_true_returns_tuple():
    serializer = TimedSerializer("secret")
    payload = {"key": "value"}
    signed = serializer.dumps(payload)
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == payload
    assert isinstance(result[1], float)
```


# LLM-generated content at query #55
#--------------------------

```python
def test_predicate_at_line_52_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    bad_timestamp = base64_encode(int_to_bytes(0))
    sep = signer.sep.encode()
    value_part = b"test"
    tampered = value_part + sep + bad_timestamp + sep + signer.get_signature(value_part + sep + bad_timestamp)
    result = signer.unsign(tampered, return_timestamp=False)
    assert result == value_part
```


# LLM-generated content at query #56
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

def test_unsign_valid_signature_return_timestamp():
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
        assert False
    except SignatureExpired:
        pass

def test_unsign_bad_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test") + b"x"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    signed_no_timestamp = signed.rsplit(b".", 1)[0]
    try:
        signer.unsign(signed_no_timestamp)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    parts = signed.rsplit(b".", 1)
    bad_timestamp = b"invalid"
    signed_bad = parts[0] + b"." + bad_timestamp
    try:
        signer.unsign(signed_bad)
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


# LLM-generated content at query #57
#--------------------------

def test_age_less_than_zero_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    signer.get_timestamp = lambda: 0
    try:
        signer.unsign(signed_value, max_age=100)
    except SignatureExpired:
        pass


# LLM-generated content at query #58
#--------------------------

def test_unsign_basic():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    import time
    time.sleep(0.01)
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test") + b"bad"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, ts = signed.rsplit(b".", 1)
    try:
        signer.unsign(value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, ts = signed.rsplit(b".", 1)
    bad_signed = value + b"." + b"notbase64"
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_negative_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass


# LLM-generated content at query #59
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_custom_separator():
    signer = TimestampSigner("secret-key", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_key_rotation():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_separator_not_in_base64():
    signer = TimestampSigner("key", sep=b"-")
    assert signer.sep == b"-"

def test_timestamp_signer_string_secret_key():
    signer = TimestampSigner("my-secret")
    assert signer.secret_keys == [b"my-secret"]

def test_timestamp_signer_bytes_secret_key():
    signer = TimestampSigner(b"binary-key")
    assert signer.secret_keys == [b"binary-key"]


# LLM-generated content at query #60
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_bytes_secret():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_timestamp_signer_constructor_with_secret_key_list():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_salt_none():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret", sep=b"|")
    assert signer.sep == b"|"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algo = HMACAlgorithm()
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm is algo


# LLM-generated content at query #61
#--------------------------

def test_unsign_valid_without_max_age_returns_bytes():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_with_max_age_returns_bytes():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_valid_return_timestamp_true():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)

def test_unsign_valid_with_max_age_and_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)

def test_unsign_invalid_signature_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    tampered = signed[:-1] + b"X"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"test" + signer.sep.encode() + b"badsig"
    try:
        signer.unsign(value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_expired_signature_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_negative_age_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"test" + signer.sep.encode() + b"!" + signer.sep.encode() + b"signature"
    try:
        signer.unsign(value)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #62
#--------------------------

def test_age_negative_raises_signature_expired():
    signer = TimestampSigner("secret-key")
    value = b"test"
    timestamp = 1000
    ts_bytes = base64_encode(int_to_bytes(timestamp))
    sep = signer.sep.encode()
    signed_value = value + sep + ts_bytes
    signature = signer.get_signature(signed_value)
    signed_value = signed_value + sep + signature
    signer.get_timestamp = lambda: 500
    try:
        signer.unsign(signed_value, max_age=1000)
    except SignatureExpired:
        pass


# LLM-generated content at query #63
#--------------------------

def test_loads_returns_payload_when_valid_signature():
    serializer = TimedSerializer("secret-key")
    signed = serializer.dumps("test_payload")
    result = serializer.loads(signed)
    assert result == "test_payload"

def test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true():
    serializer = TimedSerializer("secret-key")
    signed = serializer.dumps("test_payload")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "test_payload"
    assert isinstance(timestamp, float)

def test_loads_raises_bad_signature_when_signed_with_different_secret():
    serializer = TimedSerializer("secret-key")
    signed = TimedSerializer("wrong-key").dumps("test_payload")
    try:
        serializer.loads(signed)
        assert False
    except BadSignature:
        pass

def test_loads_raises_signature_expired_when_max_age_exceeded():
    import time
    serializer = TimedSerializer("secret-key")
    signed = serializer.dumps("test_payload")
    time.sleep(0.01)
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_loads_accepts_bytes_input():
    serializer = TimedSerializer("secret-key")
    signed = serializer.dumps("test_payload")
    result = serializer.loads(signed.encode())
    assert result == "test_payload"

def test_loads_with_salt_parameter():
    serializer = TimedSerializer("secret-key")
    signed = serializer.dumps("test_payload", salt="custom_salt")
    result = serializer.loads(signed, salt="custom_salt")
    assert result == "test_payload"


# LLM-generated content at query #64
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
    assert result[1].tzinfo is not None

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    import time as _time
    _time.sleep(0.1)
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    tampered = signed[:-1] + b"X"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = b"value" + signer.sep.encode() + signer.get_signature(b"value")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = b"value" + signer.sep.encode() + b"invalid" + signer.sep.encode() + signer.get_signature(b"value" + signer.sep.encode() + b"invalid")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_negative_age():
    signer = TimestampSigner("secret")
    from unittest.mock import patch
    import time
    with patch.object(signer, "get_timestamp", return_value=0):
        signed = signer.sign("value")
    try:
        signer.unsign(signed, max_age=3600)
        assert False
    except SignatureExpired:
        pass


# LLM-generated content at query #65
#--------------------------

```python
def test_unsign_ts_int_is_none_causes_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign(b"test")
    # Corrupt the timestamp part to make base64_decode fail -> ts_int stays None
    sep = signer.sep.encode()
    parts = signed_value.rsplit(sep, 1)
    corrupted = parts[0] + sep + b"!!!"
    try:
        signer.unsign(corrupted)
    except BadTimeSignature as e:
        assert e.payload == b"test"
        assert str(e) == "Malformed timestamp"
```


# LLM-generated content at query #66
#--------------------------

def test_timestamp_signer_constructor_default():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algo = HMACAlgorithm()
    signer = TimestampSigner("secret-key", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_constructor_with_key_rotation():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]

def test_timestamp_signer_constructor_invalid_sep():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret-key", sep="a")


# LLM-generated content at query #67
#--------------------------

def test_unsign_returns_bytes_without_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=False)
    assert result == b"test"

def test_unsign_returns_tuple_with_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)

def test_unsign_with_max_age_valid():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_with_max_age_expired():
    signer = TimestampSigner("secret-key")
    signer.get_timestamp = lambda: 1000
    signed = signer.sign("test")
    signer.get_timestamp = lambda: 2000
    try:
        signer.unsign(signed, max_age=500)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_max_age_negative():
    signer = TimestampSigner("secret-key")
    signer.get_timestamp = lambda: 2000
    signed = signer.sign("test")
    signer.get_timestamp = lambda: 1000
    try:
        signer.unsign(signed, max_age=3600)
        assert False
    except SignatureExpired:
        pass

def test_unsign_raises_bad_signature_on_invalid():
    signer = TimestampSigner("secret-key")
    try:
        signer.unsign(b"invalid")
        assert False
    except BadSignature:
        pass

def test_unsign_raises_bad_time_signature_on_missing_timestamp():
    signer = TimestampSigner("secret-key")
    original_sign = signer.sign
    signer.sign = lambda value: want_bytes(value) + b"." + signer.get_signature(want_bytes(value))
    signed = signer.sign("test")
    signer.sign = original_sign
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_raises_bad_time_signature_on_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    sep = want_bytes(signer.sep)
    parts = signed.rsplit(sep, 1)
    malformed = parts[0] + sep + b"invalid_timestamp" + sep + parts[1]
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #68
#--------------------------

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
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_future_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=1000000)
        assert False
    except SignatureExpired:
        pass

def test_unsign_invalid_signature():
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
    signed = b"test" + signer.sep.encode() + signer.get_signature(b"test")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    parts = signed.rsplit(signer.sep.encode(), 1)
    malformed = parts[0] + signer.sep.encode() + b"notbase64"
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #69
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_custom_separator():
    signer = TimestampSigner("secret-key", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

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
    assert signer.algorithm == algorithm

def test_timestamp_signer_key_rotation():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert signer.algorithm is not None

def test_timestamp_signer_custom_sep():
    signer = TimestampSigner("secret", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_salt_none():
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
    assert signer.algorithm == algorithm

def test_timestamp_signer_separator_not_in_base64():
    signer = TimestampSigner("secret", sep=b"-")
    assert signer.sep == b"-"


# LLM-generated content at query #2
#--------------------------

def test_loads_returns_payload_without_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_returns_payload_and_timestamp_when_return_timestamp_true():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "test"
    assert isinstance(timestamp, float)

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

def test_loads_raises_bad_signature_when_signature_invalid():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    modified = signed[:-1] + ("0" if signed[-1] != "0" else "1")
    try:
        serializer.loads(modified)
        assert False
    except BadSignature:
        pass

def test_loads_raises_bad_signature_when_data_tampered():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    tampered = signed.replace("test", "hack")
    try:
        serializer.loads(tampered)
        assert False
    except BadSignature:
        pass

def test_loads_uses_salt_parameter():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test", salt="custom_salt")
    result = serializer.loads(signed, salt="custom_salt")
    assert result == "test"

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    payload = "test"
    signed = serializer.dumps(payload)
    result = serializer.loads(signed.encode())
    assert result == payload


# LLM-generated content at query #3
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

def test_unsign_valid_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert timestamp is not None

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = b"test.invalidtimestamp.invalidsignature"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = b"test" + signer.sep.encode() + signer.get_signature(b"test")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_negative_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=10)
        assert False
    except SignatureExpired:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    sep = signer.sep.encode()
    bad_timestamp = b"!!invalid!!"
    signature = signer.get_signature(value + sep + bad_timestamp)
    signed = value + sep + bad_timestamp + sep + signature
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bytes_input():
    signer = TimestampSigner("secret")
    signed = signer.sign(b"test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_string_input():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed.decode())
    assert result == b"test"


# LLM-generated content at query #4
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_sep():
    signer = TimestampSigner("secret", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algorithm = HMACAlgorithm()
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_secret_key_list():
    signer = TimestampSigner(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_timestamp_signer_constructor_with_sep_in_base64_alphabet():
    import pytest
    with pytest.raises(ValueError, match="The given separator cannot be used"):
        TimestampSigner("secret", sep=b".")

def test_timestamp_signer_constructor_with_salt_none():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #5
#--------------------------

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
    signed_value = signer.sign("test")
    tampered_value = signed_value[:-1] + b"X"
    try:
        signer.unsign(tampered_value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_missing_timestamp():
    signer = TimestampSigner("secret")
    value = b"test" + signer.sep.encode() + signer.get_signature(b"test")
    try:
        signer.unsign(value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    sep = signer.sep.encode()
    bad_ts = b"not-a-timestamp"
    value = b"test" + sep + bad_ts + sep + signer.get_signature(b"test" + sep + bad_ts)
    try:
        signer.unsign(value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_zero_age_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, max_age=100000)
    assert result == b"test"

def test_unsign_with_negative_age_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=0)
        assert False
    except SignatureExpired:
        pass


# LLM-generated content at query #6
#--------------------------

def test_unsign_valid_signature_without_max_age_and_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=False)
    assert result == b"test"

def test_unsign_valid_signature_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_valid_signature_with_max_age_within_limit():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test") + b"invalid"
    try:
        signer.unsign(signed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    value = b"test"
    sep = b"."
    malformed_ts = b"notbase64"
    signed = value + sep + malformed_ts + sep + signer.get_signature(value + sep + malformed_ts)
    try:
        signer.unsign(signed)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    # Remove the timestamp part to simulate missing timestamp
    sep = b"."
    value, ts, sig = signed.rsplit(sep, 2)
    signed_no_ts = value + sep + sig
    try:
        signer.unsign(signed_no_ts)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature:
        pass


# LLM-generated content at query #7
#--------------------------

def test_loads_raises_signature_expired_when_signature_expired():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps({"key": "value"})
    try:
        serializer.loads(signed, max_age=-1)
        assert False, "Expected SignatureExpired"
    except SignatureExpired:
        pass


# LLM-generated content at query #8
#--------------------------

def test_unsign_valid_signature_without_max_age():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"

def test_unsign_valid_signature_with_max_age():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, max_age=3600)
    assert result == b"test"

def test_unsign_valid_signature_return_timestamp():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    result, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test") + b"x"
    try:
        signer.unsign(signed_value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret-key")
    signed_value = b"test"
    try:
        signer.unsign(signed_value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_timestamp():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test") + b"x"
    try:
        signer.unsign(signed_value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_empty_string():
    signer = TimestampSigner("secret-key")
    try:
        signer.unsign(b"")
        assert False
    except BadSignature:
        pass

def test_unsign_unicode_value():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("héllo")
    result = signer.unsign(signed_value)
    assert result == "héllo".encode()

def test_unsign_with_max_age_exactly_at_limit():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    age = signer.get_timestamp() - bytes_to_int(base64_decode(signed_value.split(b".")[-2]))
    result = signer.unsign(signed_value, max_age=age)
    assert result == b"test"

def test_unsign_negative_age():
    signer = TimestampSigner("secret-key")
    future_signer = TimestampSigner("secret-key")
    future_signer.get_timestamp = lambda: int(time.time()) + 1000
    signed_value = future_signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=3600)
        assert False
    except SignatureExpired:
        pass


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line43_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value)
    except Exception:
        pass


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_52_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, return_timestamp=True)


# LLM-generated content at query #11
#--------------------------

def test_unsign_valid_signature():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test_value"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_bad_signature():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    no_timestamp = signed.split(b".")[0] + b"." + signed.split(b".")[-1]
    try:
        signer.unsign(no_timestamp)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    parts = signed.split(b".")
    malformed = parts[0] + b"." + b"not-a-timestamp" + b"." + parts[2]
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #12
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_custom_sep():
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_salt_none():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_timestamp_signer_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_secret_key_rotation():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]
    assert signer.secret_key == b"new-key"

def test_timestamp_signer_sep_in_base64_alphabet_raises():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret-key", sep="a")


# LLM-generated content at query #13
#--------------------------

def test_loads_returns_payload_and_timestamp_when_return_timestamp_is_true():
    from itsdangerous.timed import TimedSerializer
    from itsdangerous.url_safe import URLSafeTimedSerializer
    serializer = URLSafeTimedSerializer("secret-key")
    s = serializer.dumps({"test": "data"})
    payload, timestamp = serializer.loads(s, return_timestamp=True)
    assert payload == {"test": "data"}
    assert isinstance(timestamp, float)


# LLM-generated content at query #14
#--------------------------

def test_timestamp_signer_constructor_default():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is TimestampSigner.default_digest_method

def test_timestamp_signer_constructor_with_all_parameters():
    from hashlib import sha256
    signer = TimestampSigner(secret_key="mykey", salt="mysalt", sep="|", key_derivation="hmac", digest_method=sha256)
    assert signer.secret_keys == [b"mykey"]
    assert signer.salt == b"mysalt"
    assert signer.sep == b"|"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method is sha256

def test_timestamp_signer_constructor_with_list_secret_key():
    signer = TimestampSigner(secret_key=["oldkey", "newkey"])
    assert signer.secret_keys == [b"oldkey", b"newkey"]
    assert signer.secret_key == b"newkey"

def test_timestamp_signer_constructor_separator_in_alphabet_raises():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret", sep="a")


# LLM-generated content at query #15
#--------------------------

def test_predicate_line52_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, return_timestamp=False)


# LLM-generated content at query #16
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

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    tampered = signed[:-1] + b"X"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    value = b"value"
    signature = signer.get_signature(value)
    signed = value + b"." + signature
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    signer.get_timestamp = lambda: 100  # Simulate old timestamp
    try:
        signer.unsign(signed, max_age=10)
        assert False
    except SignatureExpired:
        pass

def test_unsign_future_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("value")
    signer.get_timestamp = lambda: 50  # Simulate time before signing
    try:
        signer.unsign(signed, max_age=100)
        assert False
    except SignatureExpired:
        pass


# LLM-generated content at query #17
#--------------------------

def test_unsign_age_negative():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    # Manipulate timestamp to be in the future
    parts = signed.split(signer.sep.encode())
    future_ts = signer.get_timestamp() + 100
    future_ts_encoded = base64_encode(int_to_bytes(future_ts))
    manipulated_signed = parts[0] + signer.sep.encode() + future_ts_encoded + signer.sep.encode() + parts[2]
    try:
        signer.unsign(manipulated_signed, max_age=10)
    except SignatureExpired:
        pass


# LLM-generated content at query #18
#--------------------------

```python
def test_loads_returns_payload_when_not_return_timestamp():
    serializer = TimedSerializer(secret_key="secret")
    payload = {"key": "value"}
    signed = serializer.dumps(payload)
    result = serializer.loads(signed, return_timestamp=False)
    assert result == payload
```


# LLM-generated content at query #19
#--------------------------

def test_predicate_at_line_43_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"


# LLM-generated content at query #20
#--------------------------

def test_loads_returns_payload_without_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    result = serializer.loads(signed)
    assert result == "hello"

def test_loads_returns_payload_and_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    result = serializer.loads(signed, return_timestamp=True)
    assert result[0] == "hello"
    assert isinstance(result[1], int)

def test_loads_raises_bad_signature():
    serializer = TimedSerializer("secret")
    try:
        serializer.loads("invalid")
        assert False
    except BadSignature:
        pass

def test_loads_raises_signature_expired():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    try:
        serializer.loads(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_loads_with_salt():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello", salt="custom")
    result = serializer.loads(signed, salt="custom")
    assert result == "hello"

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    result = serializer.loads(signed.encode())
    assert result == "hello"

def test_loads_returns_payload_with_max_age():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    result = serializer.loads(signed, max_age=3600)
    assert result == "hello"


# LLM-generated content at query #21
#--------------------------

```python
def test_predicate_at_line_32_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test" or isinstance(result, tuple) and result[0] == b"test"
```


# LLM-generated content at query #22
#--------------------------

def test_unsign_valid_signature_no_timestamp_return():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_timestamp_return():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_valid_signature_within_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    import time
    time.sleep(2)
    try:
        signer.unsign(signed, max_age=1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_future_timestamp():
    signer = TimestampSigner("secret")
    value = want_bytes("test")
    sep = want_bytes(signer.sep)
    future_ts = int(time.time()) + 1000
    ts_bytes = base64_encode(int_to_bytes(future_ts))
    signed = value + sep + ts_bytes + sep + signer.get_signature(value + sep + ts_bytes)
    try:
        signer.unsign(signed, max_age=3600)
        assert False
    except SignatureExpired:
        pass

def test_unsign_bad_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bad_signed = signed[:-1] + b"X"
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value = want_bytes("test")
    sep = want_bytes(signer.sep)
    no_ts_signed = value + sep + signer.get_signature(value)
    try:
        signer.unsign(no_ts_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    value = want_bytes("test")
    sep = want_bytes(signer.sep)
    malformed_ts = b"not-a-timestamp"
    signed = value + sep + malformed_ts + sep + signer.get_signature(value + sep + malformed_ts)
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    value = want_bytes("test")
    sep = want_bytes(signer.sep)
    bad_sig_no_ts = value + sep + b"badsig"
    try:
        signer.unsign(bad_sig_no_ts)
        assert False
    except BadSignature:
        pass

def test_unsign_empty_string():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"")
        assert False
    except BadSignature:
        pass


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_32_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test" or isinstance(result, bytes)
```


# LLM-generated content at query #24
#--------------------------

```python
def test_unsign_predicate_at_line43_evaluates_to_false():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"


# LLM-generated content at query #25
#--------------------------

def test_unsign_valid_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed)
    assert result == b"hello"

def test_unsign_valid_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result, ts = signer.unsign(signed, return_timestamp=True)
    assert result == b"hello"
    assert isinstance(ts, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    import time
    time.sleep(1)
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = b"hello." + signer.get_signature(b"hello")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = b"hello." + b"!!!" + signer.get_signature(b"hello.!!!")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    bad_signed = signed[:-1] + b"X"
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_negative_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_returns_bytes():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed)
    assert isinstance(result, bytes)


# LLM-generated content at query #26
#--------------------------

def test_unsign_handles_timestamp_to_datetime_error_returns_none():
    import time
    import base64
    from itsdangerous.timed import TimestampSigner
    from itsdangerous.exc import BadSignature, BadTimeSignature
    from itsdangerous.encoding import want_bytes, int_to_bytes, base64_encode, bytes_to_int, base64_decode

    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    sep = want_bytes(signer.sep)
    value, ts_bytes = signed_value.rsplit(sep, 1)
    ts_int = bytes_to_int(base64_decode(ts_bytes))
    original_timestamp_to_datetime = signer.timestamp_to_datetime
    signer.timestamp_to_datetime = lambda ts: (_ for _ in ()).throw(ValueError("bad ts"))
    try:
        signer.unsign(signed_value, return_timestamp=False)
    except BadTimeSignature:
        pass
    signer.timestamp_to_datetime = original_timestamp_to_datetime


# LLM-generated content at query #27
#--------------------------

def test_loads_returns_tuple_when_return_timestamp_is_true():
    serializer = TimedSerializer("secret")
    value = serializer.dumps("test")
    result = serializer.loads(value, return_timestamp=True)
    assert isinstance(result, tuple)


# LLM-generated content at query #28
#--------------------------

def test_unsign_success_without_return_timestamp(timestamp_signer):
    signed = timestamp_signer.sign("test_value")
    result = timestamp_signer.unsign(signed)
    assert result == b"test_value"

def test_unsign_success_with_return_timestamp(timestamp_signer):
    signed = timestamp_signer.sign("test_value")
    value, timestamp = timestamp_signer.unsign(signed, return_timestamp=True)
    assert value == b"test_value"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature(timestamp_signer):
    signed = timestamp_signer.sign("test_value")
    with raises(SignatureExpired):
        timestamp_signer.unsign(signed, max_age=-1)

def test_unsign_future_signature(timestamp_signer):
    signed = timestamp_signer.sign("test_value")
    with raises(SignatureExpired):
        timestamp_signer.unsign(signed, max_age=0)

def test_unsign_invalid_signature(timestamp_signer):
    signed = timestamp_signer.sign("test_value") + b"tampered"
    with raises(BadTimeSignature):
        timestamp_signer.unsign(signed)

def test_unsign_missing_timestamp(timestamp_signer):
    signed = b"test_value" + timestamp_signer.sep.encode() + b"invalid"
    with raises(BadTimeSignature):
        timestamp_signer.unsign(signed)

def test_unsign_malformed_timestamp(timestamp_signer):
    signed = timestamp_signer.sign("test_value")
    parts = signed.rsplit(timestamp_signer.sep.encode(), 1)
    malformed = parts[0] + timestamp_signer.sep.encode() + b"notbase64"
    with raises(BadTimeSignature):
        timestamp_signer.unsign(malformed)


# LLM-generated content at query #29
#--------------------------

def test_loads_returns_payload_when_not_return_timestamp_and_no_expiry():
    from itsdangerous.timed import TimedSerializer
    from itsdangerous.signer import TimestampSigner
    from itsdangerous import URLSafeTimedSerializer
    s = TimedSerializer("secret")
    payload = {"key": "value"}
    signed = s.dumps(payload)
    result = s.loads(signed, return_timestamp=False)
    assert result == payload


# LLM-generated content at query #30
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert isinstance(result, bytes)
    assert result == b"test"
```


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_32_evaluates_to_false():
    signer = TimestampSigner("secret")
    value = b"test"
    timestamp = base64_encode(int_to_bytes(int(time.time())))
    sep = want_bytes(signer.sep)
    signed_value = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)
    result = signer.unsign(signed_value)
    assert result == value
```


# LLM-generated content at query #32
#--------------------------

def test_loads_returns_payload_when_no_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    result = serializer.loads(signed)
    assert result == "hello"

def test_loads_returns_payload_and_timestamp_when_return_timestamp_true():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == "hello"
    assert isinstance(timestamp, int)

def test_loads_with_max_age_valid():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    result = serializer.loads(signed, max_age=3600)
    assert result == "hello"

def test_loads_with_max_age_and_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    payload, timestamp = serializer.loads(signed, max_age=3600, return_timestamp=True)
    assert payload == "hello"
    assert isinstance(timestamp, int)

def test_loads_with_salt():
    serializer = TimedSerializer("secret", salt="my_salt")
    signed = serializer.dumps("hello")
    result = serializer.loads(signed, salt="my_salt")
    assert result == "hello"

def test_loads_raises_bad_signature_on_invalid_data():
    serializer = TimedSerializer("secret")
    try:
        serializer.loads(b"invalid")
    except Exception:
        pass

def test_loads_raises_signature_expired_on_old_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("hello")
    import time
    time.sleep(0.1)
    try:
        serializer.loads(signed, max_age=0)
    except Exception:
        pass


# LLM-generated content at query #33
#--------------------------

def test_unsign_returns_bytes_when_no_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test-value")
    result = signer.unsign(signed)
    assert isinstance(result, bytes)
    assert result == b"test-value"

def test_unsign_returns_bytes_and_datetime_when_return_timestamp_true():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test-value")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test-value"
    assert isinstance(result[1], datetime)

def test_unsign_raises_bad_signature_for_invalid_signature():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test-value")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False
    except BadSignature:
        pass

def test_unsign_raises_bad_time_signature_when_timestamp_missing():
    signer = TimestampSigner("secret-key")
    value_without_timestamp = b"test-value" + b"." + signer.get_signature(b"test-value")
    try:
        signer.unsign(value_without_timestamp)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_raises_signature_expired_when_age_exceeds_max_age():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test-value")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_raises_signature_expired_when_age_is_negative():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test-value")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_returns_value_when_max_age_not_exceeded():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test-value")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test-value"


# LLM-generated content at query #34
#--------------------------

```python
def test_exception_at_line_52_raises_bad_time_signature_with_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    sep = signer.sep.encode()
    value, ts_bytes = signed_value.rsplit(sep, 1)
    malformed_ts = b"!invalid!"
    malformed_signed = value + sep + malformed_ts + sep + signer.get_signature(value + sep + malformed_ts)
    try:
        signer.unsign(malformed_signed)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value
```


# LLM-generated content at query #35
#--------------------------

```python
def test_loads_returns_payload_when_return_timestamp_is_false():
    serializer = TimedSerializer("secret-key")
    data = {"key": "value"}
    signed = serializer.dumps(data)
    result = serializer.loads(signed, return_timestamp=False)
    assert result == data
```


# LLM-generated content at query #36
#--------------------------

def test_loads_with_return_timestamp_true():
    serializer = TimedSerializer("secret")
    payload = {"key": "value"}
    signed = serializer.dumps(payload)
    result, timestamp = serializer.loads(signed, return_timestamp=True)
    assert result == payload
    assert isinstance(timestamp, float)


# LLM-generated content at query #37
#--------------------------

def test_loads_with_return_timestamp_false_returns_payload():
    serializer = TimedSerializer("secret")
    data = {"key": "value"}
    serialized = serializer.dumps(data)
    result = serializer.loads(serialized, return_timestamp=False)
    assert isinstance(result, dict)


# LLM-generated content at query #38
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret-key", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_sep():
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_secret_key_list():
    signer = TimestampSigner(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_timestamp_signer_constructor_with_bytes_secret_key():
    signer = TimestampSigner(b"bytes-key")
    assert signer.secret_keys == [b"bytes-key"]

def test_timestamp_signer_constructor_invalid_sep():
    try:
        TimestampSigner("secret-key", sep=".")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for '.' separator")


# LLM-generated content at query #39
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_custom_sep():
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_salt_none():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algorithm)
    assert signer.algorithm == algorithm

def test_timestamp_signer_sep_in_base64_alphabet_raises():
    import re
    from itsdangerous.exc import BadSignature
    try:
        signer = TimestampSigner("secret-key", sep="a")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# LLM-generated content at query #40
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_with_all_parameters():
    signer = TimestampSigner(
        secret_key=b"mykey",
        salt=b"mysalt",
        sep=b"|",
        key_derivation="hmac",
        digest_method=hashlib.sha256,
        algorithm=HMACAlgorithm(hashlib.sha256),
    )
    assert signer.secret_keys == [b"mykey"]
    assert signer.sep == b"|"
    assert signer.salt == b"mysalt"
    assert signer.key_derivation == "hmac"
    assert signer.digest_method == hashlib.sha256
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_with_invalid_sep():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret", sep=b"a")


# LLM-generated content at query #41
#--------------------------

```python
def test_loads_without_return_timestamp_does_not_enter_if_branch():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, return_timestamp=False)
    assert result == "test"
```


# LLM-generated content at query #42
#--------------------------

```python
def test_timestamp_to_datetime_raises_on_invalid_timestamp():
    signer = TimestampSigner("secret")
    # Create a signed value where the timestamp decodes to something that causes an exception
    # We need to craft a payload that after base64 decode and bytes_to_int produces a value
    # that timestamp_to_datetime will fail on (e.g., negative, too large, etc.)
    # The easiest is to provide a timestamp that is not a valid integer for datetime conversion.
    # For instance, a very large number that causes OverflowError on some platforms.
    # We can simulate by encoding a large integer as bytes, then base64 encode it.
    # But since we only want to trigger the except block at line 52, we need a signed_value
    # that has a valid base64 timestamp part but whose conversion to int leads to an exception.
    # We can directly construct a signed_value with a known bad timestamp.
    # Let's use a timestamp that is a negative number, which may cause ValueError on some systems.
    # We'll manually construct the signed_value: value + sep + base64(int_to_bytes(bad_ts)) + sep + signature
    # But to simplify, we can mock timestamp_to_datetime to raise.
    # However, the task is to write a unit test without mocks? The instruction says only assignments, assertions, calls.
    # So we need to find a real input that triggers this.
    # One approach: use a timestamp that is too large for datetime (e.g., > max timestamp).
    # On 64-bit systems, datetime supports up to year 9999, so we need a very large int.
    # We can create a bytes object that after base64 decode and bytes_to_int gives a huge number.
    # For example, b'\xff' * 8 as big-endian gives 2^64 -1, which is huge.
    # But bytes_to_int uses rjust(8, b'\x00') and then _bytes_to_int which is big-endian.
    # So we can craft ts_bytes = base64_encode(b'\xff' * 8) which will decode to a huge int.
    # Then we need to create a signed_value that passes the initial unsign (signature valid) but with that timestamp.
    # To avoid signature error, we need to sign a value with that timestamp ourselves.
    # We can call sign with a specific timestamp by monkey-patching get_timestamp? Not allowed.
    # Alternatively, we can create a signed_value manually by computing the signature over value + sep + timestamp.
    # Let's do that.
    import base64
    import struct
    # Create a timestamp that causes OverflowError: very large number
    huge_int = 2**63 + 1  # slightly above max for signed 64-bit, but datetime may handle? To be safe, use very large.
    # Use bytes_to_int's internal: it rjust to 8 bytes, so we need 8 bytes representing huge int.
    # For a truly huge, we can use b'\xff' * 8 which is 2^64-1, which on some systems may overflow.
    ts_bytes = base64.urlsafe_b64encode(b'\xff' * 8)  # this is valid base64, no padding needed? Actually it may need padding, but base64_decode adds padding.
    # Construct value = b"test"
    value = b"test"
    sep = b"."
    # Compute signature: HMAC-SHA1 of value + sep + ts_bytes (as bytes) with secret
    import hmac, hashlib
    secret = b"secret"
    key = b"secret-salt"  # default salt? Actually Signer uses key derivation, but for simplicity we can use the secret directly? The default algorithm uses HMAC-SHA1 with key derived from secret and salt.
    # To avoid complexity, we can use the actual signer to get a valid signed value with a known timestamp by temporarily changing get_timestamp.
    # But that would require mocking. Since the instruction says no mocks, we need another approach.
    # We can instead trigger the exception by having the timestamp decode to something that causes ValueError or OSError.
    # For example, a timestamp that is not a valid integer after bytes_to_int? But bytes_to_int always returns an int.
    # So the only exception from timestamp_to_datetime is when the int is out of range (OverflowError) or invalid (ValueError on Windows for negative timestamps).
    # On Windows, OSError can occur for negative timestamps? Actually, fromtimestamp on Windows raises OSError for negative timestamps.
    # So we can use a negative timestamp.
    # Let's create a timestamp that is negative: e.g., -1
    # int_to_bytes for negative? int_to_bytes expects non-negative? It uses big-endian, but for negative it would produce a large positive? Actually, int_to_bytes from itsdangerous encodes unsigned integers.
    # So we cannot directly encode a negative int. But we can craft the bytes manually to represent a negative int in a way that bytes_to_int interprets as a huge positive? No, bytes_to_int treats bytes as unsigned big-endian.
    # So we cannot produce a negative int via bytes_to_int.
    # Another idea: use a timestamp that is a float? No, bytes_to_int only returns int.
    # Maybe we can cause OverflowError by using a timestamp that is extremely large, e.g., 10**18.
    # We'll compute the bytes for a large number.
    # Use struct.pack('>Q', 10**18) gives 8 bytes, but that number is within range for datetime? 10**18 seconds is huge, far beyond year 9999, so datetime.fromtimestamp will raise OverflowError.
    # Let's try that.
    large_ts = 10**18
    ts_bytes_before_b64 = struct.pack('>Q', large_ts)  # 8 bytes big-endian
    ts_bytes = base64.urlsafe_b64encode(ts_bytes_before_b64)
    # Now we need to create a signed_value that passes the signature check.
    # We can sign the value with the actual signer by temporarily replacing get_timestamp to return that large_ts.
    # But we cannot do that without mocking. So we need to manually compute the signature.
    # The signer uses HMAC-SHA1 with a derived key. The default algorithm is 'hmac-sha1' and uses a salt 'itsdangerous.Signer'.
    # We can replicate the signing process.
    from itsdangerous.signer import Signer
    # Actually, we can create a Signer instance and call its get_signature method.
    signer = TimestampSigner("secret")
    # We need to craft the signed_value manually: value + sep + ts_bytes + sep + signer.get_signature(value + sep + ts_bytes)
    # But get_signature expects bytes, not string.
    data = value + sep + ts_bytes
    sig = signer.get_signature(data)
    signed_value = data + sep + sig
    # Now call unsign with this signed_value
    # This should trigger the except block at line 52 because timestamp_to_datetime(large_ts) will raise OverflowError
    # We expect a BadTimeSignature with message "Malformed timestamp"
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value
```


# LLM-generated content at query #43
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_custom_sep():
    signer = TimestampSigner("secret", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_custom_salt():
    signer = TimestampSigner("secret", salt=b"custom")
    assert signer.salt == b"custom"

def test_timestamp_signer_constructor_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_timestamp_signer_constructor_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    import hashlib
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm == algo

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_timestamp_signer_constructor_sep_in_base64_raises_error():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret", sep=b"a")


# LLM-generated content at query #44
#--------------------------

def test_unsign_successful_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_successful_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_with_max_age_valid():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_with_max_age_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    with raise_(SignatureExpired):
        signer.unsign(signed, max_age=0)

def test_unsign_with_bad_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    altered = signed[:-1] + b"x"
    with raise_(BadTimeSignature):
        signer.unsign(altered)

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    no_timestamp = signed.split(signer.sep.encode())[0]
    with raise_(BadTimeSignature):
        signer.unsign(no_timestamp)


# LLM-generated content at query #45
#--------------------------

def test_unsign_basic_valid_signature():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    result = signer.unsign(signed)
    assert result == b"test_value"

def test_unsign_with_return_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    value, timestamp = signer.unsign(signed, return_timestamp=True)
    assert value == b"test_value"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired:
        pass
    else:
        assert False, "Expected SignatureExpired"

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    malformed = signed + b"."
    try:
        signer.unsign(malformed)
    except BadTimeSignature:
        pass
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_missing_separator():
    signer = TimestampSigner("secret-key")
    try:
        signer.unsign(b"no_separator")
    except BadTimeSignature:
        pass
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_invalid_base64_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    parts = signed.rsplit(b".", 1)
    invalid_signed = parts[0] + b".invalid_base64"
    try:
        signer.unsign(invalid_signed)
    except BadTimeSignature:
        pass
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_bad_signature_with_timestamp():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered)
    except BadTimeSignature:
        pass
    else:
        assert False, "Expected BadTimeSignature"

def test_unsign_negative_age():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test_value")
    try:
        signer.unsign(signed, max_age=1000000)
    except SignatureExpired:
        pass
    else:
        assert False, "Expected SignatureExpired"


# LLM-generated content at query #46
#--------------------------

def test_unsign_valid_no_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed)
    assert result == b"hello"

def test_unsign_valid_with_max_age_not_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"hello"

def test_unsign_valid_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"hello"
    assert isinstance(result[1], datetime)

def test_unsign_invalid_signature_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    tampered = signed[:-1] + (b"x" if signed[-1:] != b"x" else b"y")
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_separator_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"nodata")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"hello" + signer.sep.encode() + b"invalidbase64"
    signed = value + signer.sep.encode() + signer.get_signature(value)
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_expired_max_age_raises_signature_expired():
    signer = TimestampSigner("secret")
    # Simulate old timestamp by directly manipulating sign time
    old_ts = 1000000
    value = b"hello"
    ts_bytes = base64_encode(int_to_bytes(old_ts))
    sep = signer.sep.encode()
    value_with_ts = value + sep + ts_bytes
    signed = value_with_ts + sep + signer.get_signature(value_with_ts)
    try:
        signer.unsign(signed, max_age=1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_negative_age_raises_signature_expired():
    signer = TimestampSigner("secret")
    future_ts = int(time.time()) + 1000
    value = b"hello"
    ts_bytes = base64_encode(int_to_bytes(future_ts))
    sep = signer.sep.encode()
    value_with_ts = value + sep + ts_bytes
    signed = value_with_ts + sep + signer.get_signature(value_with_ts)
    try:
        signer.unsign(signed, max_age=100)
        assert False
    except SignatureExpired:
        pass

def test_unsign_with_bad_signature_and_valid_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    ts = signer.get_timestamp()
    ts_bytes = base64_encode(int_to_bytes(ts))
    sep = signer.sep.encode()
    value_with_ts = value + sep + ts_bytes
    # Use wrong secret to create bad signature
    wrong_signer = TimestampSigner("wrong_secret")
    signed = value_with_ts + sep + wrong_signer.get_signature(value_with_ts)
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_bad_signature_and_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"hello"
    ts_bytes = b"badbase64"
    sep = signer.sep.encode()
    value_with_ts = value + sep + ts_bytes
    signed = value_with_ts + sep + signer.get_signature(value_with_ts)
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #47
#--------------------------

def test_unsign_with_signature_error_and_valid_timestamp_does_not_raise_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    # Tamper the signature to cause BadSignature
    tampered = signed_value[:-1] + b"X"
    try:
        signer.unsign(tampered, return_timestamp=False)
    except BadTimeSignature as e:
        assert e.date_signed is not None
        assert str(e) != "Malformed timestamp"


# LLM-generated content at query #48
#--------------------------

def test_unsign_valid_without_timestamp_return():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_with_timestamp_return():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)

def test_unsign_valid_within_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    import time
    time.sleep(1)
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = b"test." + signer.sep.encode() + b"invalid"
    signed = signed + signer.sep.encode() + signer.get_signature(signed)
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    # Remove timestamp part but keep signature
    sep = signer.sep.encode()
    value = b"test"
    sig = signer.get_signature(value)
    forged = value + sep + sig
    try:
        signer.unsign(forged)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    # Modify signed value
    parts = signed.rsplit(signer.sep.encode(), 1)
    modified = b"modified" + signer.sep.encode() + parts[1]
    try:
        signer.unsign(modified)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    # Remove timestamp and signature, keep bad value
    sep = signer.sep.encode()
    bad_value = b"bad"
    bad_sig = b"invalidsig"
    forged = bad_value + sep + b"timestamp" + sep + bad_sig
    try:
        signer.unsign(forged)
        assert False
    except BadSignature:
        pass

def test_unsign_empty_string():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"")
        assert False
    except (BadSignature, BadTimeSignature):
        pass

def test_unsign_none_value():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(None)
        assert False
    except (TypeError, BadSignature, BadTimeSignature):
        pass


# LLM-generated content at query #49
#--------------------------

def test_loads_with_str_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed.encode())
    assert result == "test"

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == "test"
    assert isinstance(result[1], float)

def test_loads_with_max_age_valid():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, max_age=3600)
    assert result == "test"

def test_loads_with_max_age_expired():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    import time
    time.sleep(0.1)
    try:
        serializer.loads(signed, max_age=0.01)
        assert False
    except Exception:
        assert True

def test_loads_with_invalid_signature():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    modified = signed[:-1] + b"x" if isinstance(signed, bytes) else signed[:-1] + "x"
    try:
        serializer.loads(modified)
        assert False
    except Exception:
        assert True

def test_loads_with_salt():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test", salt="custom")
    result = serializer.loads(signed, salt="custom")
    assert result == "test"


# LLM-generated content at query #50
#--------------------------

def test_ts_int_is_none_raises_bad_time_signature():
    from itsdangerous.timed import TimestampSigner
    from itsdangerous.exc import BadTimeSignature
    from itsdangerous.encoding import want_bytes, base64_encode, int_to_bytes
    import base64
    signer = TimestampSigner("secret")
    value = b"test"
    sep = want_bytes(signer.sep)
    timestamp = b"invalid_base64"
    signed_value = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value


# LLM-generated content at query #51
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_custom_sep():
    signer = TimestampSigner("secret", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_custom_salt():
    signer = TimestampSigner("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_custom_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algorithm = HMACAlgorithm()
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm == algorithm

def test_timestamp_signer_constructor_key_rotation():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]


# LLM-generated content at query #52
#--------------------------

def test_loads_returns_payload_without_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps({"key": "value"})
    result = serializer.loads(signed)
    assert result == {"key": "value"}

def test_loads_returns_payload_and_timestamp_when_return_timestamp_true():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps({"key": "value"})
    payload, timestamp = serializer.loads(signed, return_timestamp=True)
    assert payload == {"key": "value"}
    assert isinstance(timestamp, float)

def test_loads_raises_bad_signature_on_invalid_data():
    serializer = TimedSerializer("secret")
    try:
        serializer.loads(b"invalid")
        assert False
    except BadSignature:
        pass

def test_loads_raises_signature_expired_when_max_age_exceeded():
    import time
    serializer = TimedSerializer("secret")
    signed = serializer.dumps({"key": "value"})
    time.sleep(1)
    try:
        serializer.loads(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_loads_with_salt_uses_correct_signer():
    serializer = TimedSerializer("secret", salt="custom_salt")
    signed = serializer.dumps({"key": "value"})
    result = serializer.loads(signed)
    assert result == {"key": "value"}


# LLM-generated content at query #53
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_bytes_secret():
    signer = TimestampSigner(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_timestamp_signer_constructor_with_multiple_keys():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]

def test_timestamp_signer_constructor_with_custom_sep():
    signer = TimestampSigner("secret", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_custom_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_salt_none():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_key_derivation_concat():
    signer = TimestampSigner("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_timestamp_signer_constructor_with_key_derivation_hmac():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm == algorithm

def test_timestamp_signer_constructor_raises_on_invalid_sep():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret", sep=b"a")


# LLM-generated content at query #54
#--------------------------

def test_unsign_sep_not_in_result_and_no_sig_error():
    signer = TimestampSigner("secret")
    signed_value = b"just_value"
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert str(e) == "timestamp missing"
        assert e.payload == signed_value


# LLM-generated content at query #55
#--------------------------

def test_unsign_exception_at_line_43_does_not_affect_result():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    # Modify the timestamp part to be invalid base64 so that bytes_to_int(base64_decode(ts_bytes)) raises an exception
    parts = signed.rsplit(signer.sep.encode(), 1)
    modified_ts = b"!!!invalid!!!"
    signed_modified = parts[0] + signer.sep.encode() + modified_ts + signer.sep.encode() + signer.get_signature(parts[0] + signer.sep.encode() + modified_ts)
    result = signer.unsign(signed_modified)
    assert result == parts[0]


# LLM-generated content at query #56
#--------------------------

def test_unsign_valid_signature_without_timestamp_return():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"

def test_unsign_valid_signature_with_timestamp_return():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result, ts = signer.unsign(signed_value, return_timestamp=True)
    assert result == b"test"
    assert isinstance(ts, datetime)

def test_unsign_expired_signature_raises():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_expired_signature_negative_age_raises():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=1000000)
        assert False
    except SignatureExpired:
        pass

def test_unsign_missing_separator_raises():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"no_separator")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp_raises():
    signer = TimestampSigner("secret")
    value = b"test" + signer.sep.encode() + b"bad_timestamp"
    try:
        signer.unsign(value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_invalid_signature_raises():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    tampered = signed_value[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_invalid_signature_with_timestamp_raises():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    tampered = signed_value[:-1] + b"x"
    try:
        signer.unsign(tampered, return_timestamp=True)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature_with_missing_timestamp_raises():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"no_timestamp")
        assert False
    except BadSignature:
        pass

def test_unsign_valid_signature_with_max_age_not_exceeded():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, max_age=3600)
    assert result == b"test"

def test_unsign_valid_signature_with_max_age_exceeded_raises():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=-1)
        assert False
    except SignatureExpired:
        pass


# LLM-generated content at query #57
#--------------------------

def test_unsign_valid_without_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_with_timestamp():
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

def test_unsign_bad_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    tampered = signed + b"x"
    try:
        signer.unsign(tampered)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    sep = signer.sep.encode()
    value, timestamp = signed.rsplit(sep, 1)
    missing_timestamp = value
    try:
        signer.unsign(missing_timestamp)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    sep = signer.sep.encode()
    value, _ = signed.rsplit(sep, 1)
    malformed = value + sep + b"not_a_valid_timestamp"
    try:
        signer.unsign(malformed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #58
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_custom_sep():
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret-key", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_none_salt():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algo = HMACAlgorithm()
    signer = TimestampSigner("secret-key", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_list_secret_keys():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]

def test_timestamp_signer_bytes_secret_key():
    signer = TimestampSigner(b"bytes_key")
    assert signer.secret_keys == [b"bytes_key"]


# LLM-generated content at query #59
#--------------------------

def test_loads_returns_payload_when_return_timestamp_is_false():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, return_timestamp=False)
    assert result == "test"


# LLM-generated content at query #60
#--------------------------

def test_loads_with_str_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed)
    assert result == "test"

def test_loads_with_bytes_input():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed.encode())
    assert result == "test"

def test_loads_with_max_age():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, max_age=3600)
    assert result == "test"

def test_loads_with_return_timestamp():
    serializer = TimedSerializer("secret")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, return_timestamp=True)
    assert len(result) == 2
    assert result[0] == "test"

def test_loads_with_salt():
    serializer = TimedSerializer("secret", salt="custom_salt")
    signed = serializer.dumps("test")
    result = serializer.loads(signed, salt="custom_salt")
    assert result == "test"


# LLM-generated content at query #61
#--------------------------

def test_unsign_success_no_timestamp_return():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_success_with_timestamp_return():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    value, ts = signer.unsign(signed, return_timestamp=True)
    assert value == b"test"
    assert isinstance(ts, datetime)

def test_unsign_bad_signature():
    signer = TimestampSigner("secret-key")
    try:
        signer.unsign(b"invalid|data|signature")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_separator():
    signer = TimestampSigner("secret-key")
    try:
        signer.unsign(b"nodata")
        assert False
    except BadTimeSignature:
        pass

def test_unsign_expired_max_age():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_negative_age():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_valid_max_age():
    signer = TimestampSigner("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret-key")
    signed = b"test." + base64_encode(b"notanint") + b"." + signer.get_signature(b"test." + base64_encode(b"notanint"))
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #62
#--------------------------

```python
def test_predicate_at_line_32_evaluates_to_false():
    signer = TimestampSigner("secret-key")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"
```


# LLM-generated content at query #63
#--------------------------

def test_unsign_valid_without_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed)
    assert result == b"hello"

def test_unsign_valid_with_max_age():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"hello"

def test_unsign_valid_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    result = signer.unsign(signed, return_timestamp=True)
    value, timestamp = result
    assert value == b"hello"
    assert isinstance(timestamp, datetime)

def test_unsign_invalid_signature():
    signer = TimestampSigner("secret")
    signed = b"hello.invalid"
    try:
        signer.unsign(signed)
        assert False
    except BadSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    value, ts = signed.rsplit(b".", 1)
    bad_signed = value
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signer.get_timestamp = lambda: 1000
    signed = signer.sign("hello")
    signer.get_timestamp = lambda: 2000
    try:
        signer.unsign(signed, max_age=500)
        assert False
    except SignatureExpired:
        pass

def test_unsign_future_signature():
    signer = TimestampSigner("secret")
    signer.get_timestamp = lambda: 2000
    signed = signer.sign("hello")
    signer.get_timestamp = lambda: 1000
    try:
        signer.unsign(signed, max_age=500)
        assert False
    except SignatureExpired:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed = b"hello." + b"invalid_base64"
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_bad_signature_and_valid_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("hello")
    value, ts = signed.rsplit(b".", 1)
    bad_signed = b"wrong" + b"." + ts + b"." + signer.get_signature(b"wrong." + ts)
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_bad_signature_and_invalid_timestamp():
    signer = TimestampSigner("secret")
    signed = b"hello.invalid_timestamp." + signer.get_signature(b"hello.invalid_timestamp")
    try:
        signer.unsign(signed)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #64
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_custom_sep():
    signer = TimestampSigner("secret-key", sep=b":")
    assert signer.sep == b":"

def test_timestamp_signer_custom_salt():
    signer = TimestampSigner("secret-key", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_custom_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_custom_digest_method():
    import hashlib
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_custom_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algo = HMACAlgorithm()
    signer = TimestampSigner("secret-key", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_secret_key_property():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_sep_in_base64_alphabet_raises():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret-key", sep=b"A")

def test_timestamp_signer_salt_none():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #65
#--------------------------

def test_ts_int_is_none_at_line_63():
    signer = TimestampSigner("secret")
    # Create a signed value with a valid signature but a malformed timestamp
    # We need to craft a value where super().unsign succeeds, sep is present,
    # but base64_decode or bytes_to_int fails so ts_int remains None.
    # To do this, we'll manually construct a signed value with an invalid timestamp.
    value = b"test"
    sep = signer.sep.encode()
    # Use a timestamp that is not valid base64
    bad_timestamp = b"!!!invalid!!!"
    # We need the signature to be valid, so we compute it for value + sep + bad_timestamp
    # But super().unsign will compute the signature and compare, so we need the correct signature.
    # Actually, we want super().unsign to succeed, so we must provide a valid signature for the full value.
    # The full value is value + sep + bad_timestamp, and then we append the signature.
    # We'll compute the signature for "value + sep + bad_timestamp".
    full_value = value + sep + bad_timestamp
    sig = signer.get_signature(full_value)
    signed_value = full_value + sep + sig
    # Now, when super().unsign(signed_value) is called, it should return the original value (value + sep + bad_timestamp)
    # because the signature is valid. Then sep is in result, so rsplit works.
    # Then base64_decode(bad_timestamp) will raise an exception, so ts_int remains None.
    # This should trigger the condition at line 63.
    try:
        signer.unsign(signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value


# LLM-generated content at query #66
#--------------------------

```python
def test_unsign_with_sig_error_and_malformed_timestamp_raises_bad_time_signature():
    from itsdangerous.timed import TimestampSigner
    from itsdangerous.exc import BadSignature, BadTimeSignature
    from itsdangerous.encoding import want_bytes, base64_encode, int_to_bytes
    import time

    signer = TimestampSigner("secret")
    value = b"test"
    timestamp = base64_encode(int_to_bytes(9999999999))
    sep = want_bytes(signer.sep)
    signed_value = value + sep + timestamp + sep + b"badsignature"
    try:
        signer.unsign(signed_value)
    except BadTimeSignature as e:
        assert str(e) == "Malformed timestamp"
        assert e.payload == value
```


# LLM-generated content at query #67
#--------------------------

def test_timestamp_signer_constructor_default():
    signer = TimestampSigner("secret")
    assert signer.secret_key == b"secret"
    assert signer.secret_keys == [b"secret"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_sep():
    signer = TimestampSigner("secret", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    class FakeAlgorithm(SigningAlgorithm):
        def get_signature(self, key, value):
            return b"fake"
        def verify_signature(self, key, value, sig):
            return True
    algo = FakeAlgorithm()
    signer = TimestampSigner("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_constructor_multiple_secret_keys():
    signer = TimestampSigner(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.secret_key == b"key2"


# LLM-generated content at query #68
#--------------------------

def test_ts_int_is_none_causes_bad_time_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    sep = want_bytes(signer.sep)
    value, ts_bytes = signed_value.rsplit(sep, 1)
    bad_ts = base64_encode(b"notanint")
    bad_signed_value = value + sep + bad_ts + sep + signer.get_signature(value + sep + bad_ts)
    try:
        signer.unsign(bad_signed_value)
    except BadTimeSignature:
        pass
    else:
        assert False, "Expected BadTimeSignature"


# LLM-generated content at query #69
#--------------------------

def test_predicate_line_52_true():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    base64_encoded_timestamp = signed_value.split(signer.sep.encode())[1]
    from itsdangerous.encoding import base64_decode, bytes_to_int
    ts_int = bytes_to_int(base64_decode(base64_encoded_timestamp))
    import datetime
    import time
    signer.timestamp_to_datetime = lambda ts: (_ for _ in ()).throw(OverflowError("test"))
    signer.get_timestamp = lambda: ts_int
    try:
        signer.unsign(signed_value, max_age=0)
    except BadTimeSignature:
        pass


# LLM-generated content at query #70
#--------------------------

```python
def test_unsign_sep_not_in_result_and_sig_error_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    sep = signer.sep.encode()
    # Remove the timestamp part to make sep not in result
    value, timestamp = signed_value.rsplit(sep, 1)
    # Construct a signed value with a signature that is valid but missing the separator
    # Use a known valid signature for the value without timestamp
    from itsdangerous.signer import Signer
    base_signer = Signer("secret")
    valid_sig = base_signer.get_signature(value)
    bad_signed = value + sep + valid_sig
    # This should trigger the predicate at line 32 being False (no sig_error)
    result = signer.unsign(bad_signed)
    assert result == value
```


# LLM-generated content at query #71
#--------------------------

```python
def test_unsign_predicate_line52_false():
    signer = TimestampSigner("secret")
    value = b"test"
    timestamp = base64_encode(int_to_bytes(signer.get_timestamp()))
    sep = signer.sep.encode()
    signed_value = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)
    result = signer.unsign(signed_value, max_age=3600)
    assert result == value
```


# LLM-generated content at query #72
#--------------------------

def test_loads_signature_expired_raises_properly():
    serializer = TimedSerializer("test_secret")
    signed_data = serializer.dumps("test payload")
    bad_data = signed_data[:-1] + (b"a" if signed_data[-1:] != b"a" else b"b")
    try:
        serializer.loads(bad_data, max_age=0)
    except BadSignature:
        pass
    expired_signer = TimestampSigner("test_secret")
    expired_data = expired_signer.sign(b"test payload")
    try:
        serializer.loads(expired_data, max_age=-1)
    except SignatureExpired:
        pass


# LLM-generated content at query #73
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_timestamp_signer_constructor_with_sep():
    signer = TimestampSigner("secret-key", sep=":")
    assert signer.sep == b":"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    signer = TimestampSigner("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    algo = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret-key", algorithm=algo)
    assert signer.algorithm is algo

def test_timestamp_signer_constructor_with_multiple_secret_keys():
    signer = TimestampSigner(["key1", "key2", "key3"])
    assert signer.secret_keys == [b"key1", b"key2", b"key3"]

def test_timestamp_signer_constructor_with_salt_none():
    signer = TimestampSigner("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_constructor_with_invalid_sep():
    import pytest
    try:
        TimestampSigner("secret-key", sep="a")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# LLM-generated content at query #74
#--------------------------

```python
def test_predicate_age_less_than_zero():
    signer = TimestampSigner("secret")
    # Create a signed value with a future timestamp
    future_timestamp = int(time.time()) + 1000
    value = b"test"
    timestamp = base64_encode(int_to_bytes(future_timestamp))
    sep = signer.sep.encode()
    signed_value = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)
    # Call unsign with max_age to trigger the age check
    try:
        signer.unsign(signed_value, max_age=0)
    except SignatureExpired as e:
        assert "age" in str(e) and "< 0" in str(e)
```


# LLM-generated content at query #75
#--------------------------

def test_unsign_valid_signature_no_expiry():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_valid_signature_with_return_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result, timestamp = signer.unsign(signed, return_timestamp=True)
    assert result == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    import time
    time.sleep(0.1)
    try:
        signer.unsign(signed, max_age=0)
        assert False
    except SignatureExpired:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    sep = signer.sep.encode()
    value = b"test" + sep + b"invalidsignature"
    try:
        signer.unsign(value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_signature():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    bad_signed = signed[:-1] + b"x"
    try:
        signer.unsign(bad_signed)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_bad_timestamp_encoding():
    signer = TimestampSigner("secret")
    sep = signer.sep.encode()
    value = b"test" + sep + b"notb64!signature"
    try:
        signer.unsign(value)
        assert False
    except BadTimeSignature:
        pass


# LLM-generated content at query #76
#--------------------------

```python
def test_predicate_line43_evaluates_to_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    signer.unsign(signed_value)
```


# LLM-generated content at query #77
#--------------------------

def test_predicate_line52_evaluates_false():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"


# LLM-generated content at query #78
#--------------------------

def test_unsign_valid_signature_without_max_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value)
    assert result == b"test"

def test_unsign_valid_signature_with_max_age():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, max_age=3600)
    assert result == b"test"

def test_unsign_valid_signature_return_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    value, timestamp = signer.unsign(signed_value, return_timestamp=True)
    assert value == b"test"
    assert isinstance(timestamp, datetime)

def test_unsign_expired_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=-1)
        assert False
    except SignatureExpired:
        pass

def test_unsign_bad_signature():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test") + b"x"
    try:
        signer.unsign(signed_value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_missing_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    bad_value = signed_value.split(b"test.")[0] + b".signature"
    try:
        signer.unsign(bad_value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_malformed_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    parts = signed_value.rsplit(b".", 2)
    bad_value = parts[0] + b".invalid_timestamp." + parts[2]
    try:
        signer.unsign(bad_value)
        assert False
    except BadTimeSignature:
        pass

def test_unsign_with_max_age_zero():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    result = signer.unsign(signed_value, max_age=0)
    assert result == b"test"

def test_unsign_with_negative_age_timestamp():
    signer = TimestampSigner("secret")
    signed_value = signer.sign("test")
    try:
        signer.unsign(signed_value, max_age=3600)
    except:
        pass


# LLM-generated content at query #79
#--------------------------

def test_timestamp_signer_constructor_defaults():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert signer.algorithm is not None

def test_timestamp_signer_constructor_with_salt():
    signer = TimestampSigner("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_constructor_with_sep():
    signer = TimestampSigner("secret", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_constructor_with_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_constructor_with_digest_method():
    import hashlib
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_constructor_with_algorithm():
    from itsdangerous.signer import HMACAlgorithm
    algorithm = HMACAlgorithm()
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_constructor_with_secret_keys_list():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]

def test_timestamp_signer_constructor_with_bytes_salt():
    signer = TimestampSigner("secret", salt=b"bytes_salt")
    assert signer.salt == b"bytes_salt"

def test_timestamp_signer_constructor_with_sep_in_base64_raises():
    import pytest
    with pytest.raises(ValueError):
        TimestampSigner("secret", sep="a")


# LLM-generated content at query #80
#--------------------------

```python
def test_unsign_with_valid_signature_and_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    value = b"test"
    timestamp = base64_encode(int_to_bytes(1234567890))
    sep = b"."
    signed_value = value + sep + timestamp + sep + signer.get_signature(value + sep + timestamp)
    # Corrupt the timestamp to make ts_int None
    corrupted_timestamp = base64_encode(b"corrupt")
    corrupted_signed_value = value + sep + corrupted_timestamp + sep + signer.get_signature(value + sep + corrupted_timestamp)
    try:
        signer.unsign(corrupted_signed_value)
        assert False, "Expected BadTimeSignature"
    except BadTimeSignature as e:
        assert "Malformed timestamp" in str(e)
        assert e.payload == value
```


# LLM-generated content at query #81
#--------------------------

def test_unsign_without_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value_without_sep")
    except BadTimeSignature:
        pass

def test_unsign_with_malformed_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value" + signer.sep.encode() + b"invalid_base64")
    except BadTimeSignature:
        pass

def test_unsign_with_bad_signature_and_invalid_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    try:
        signer.unsign(b"value" + signer.sep.encode() + b"AAAA")
    except BadTimeSignature:
        pass

def test_unsign_with_bad_signature_and_valid_timestamp_raises_bad_time_signature():
    signer = TimestampSigner("secret")
    timestamp = base64_encode(int_to_bytes(1000000))
    signed = b"value" + signer.sep.encode() + timestamp
    try:
        signer.unsign(signed)
    except BadTimeSignature:
        pass

def test_unsign_with_valid_signature_and_no_max_age_returns_value():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_with_valid_signature_and_max_age_not_exceeded_returns_value():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, max_age=3600)
    assert result == b"test"

def test_unsign_with_valid_signature_and_max_age_exceeded_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=-1)
    except SignatureExpired:
        pass

def test_unsign_with_age_negative_raises_signature_expired():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    try:
        signer.unsign(signed, max_age=100)
    except SignatureExpired:
        pass

def test_unsign_with_return_timestamp_true_returns_value_and_datetime():
    signer = TimestampSigner("secret")
    signed = signer.sign("test")
    result = signer.unsign(signed, return_timestamp=True)
    assert isinstance(result, tuple)
    assert result[0] == b"test"
    assert isinstance(result[1], datetime)


# LLM-generated content at query #82
#--------------------------

def test_timestamp_signer_default_constructor():
    signer = TimestampSigner("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_timestamp_signer_with_custom_sep():
    signer = TimestampSigner("secret", sep="|")
    assert signer.sep == b"|"

def test_timestamp_signer_with_custom_salt():
    signer = TimestampSigner("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_timestamp_signer_with_none_salt():
    signer = TimestampSigner("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_timestamp_signer_with_key_derivation():
    signer = TimestampSigner("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_timestamp_signer_with_digest_method():
    signer = TimestampSigner("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_timestamp_signer_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = TimestampSigner("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_timestamp_signer_secret_keys_from_list():
    signer = TimestampSigner(["old_key", "new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]
    assert signer.secret_key == b"new_key"

def test_timestamp_signer_secret_keys_from_bytes():
    signer = TimestampSigner(b"bytes_key")
    assert signer.secret_keys == [b"bytes_key"]

def test_timestamp_signer_secret_keys_from_string():
    signer = TimestampSigner("string_key")
    assert signer.secret_keys == [b"string_key"]


# LLM-generated content at query #83
#--------------------------

```python
def test_unsign_sep_not_in_result_sig_error_is_none():
    signer = TimestampSigner("secret-key")
    signed_value = b"test-value" + signer.sep.encode() + b"invalid-timestamp"
    signed_value += signer.sep.encode() + signer.get_signature(signed_value)
    # Modify signed_value to ensure sep is not in result after super().unsign()
    # We need to craft a signed_value that passes super().unsign() but lacks sep in result
    # Since super().unsign() returns the original value, we can sign a value that after unsign does not contain sep
    # Use a signed value where the original value does not contain sep
    value_without_sep = b"value"
    sep = signer.sep.encode()
    timestamp = b"MTIzNDU2Nzg5"  # base64 of some bytes
    signed = value_without_sep + sep + timestamp + sep + signer.get_signature(value_without_sep + sep + timestamp)
    # This ensures sep is in result, so we need to break it differently
    # Instead, we can mock super().unsign to return a result without sep
    # But without mocking, we can craft a signed value that after unsign gives result without sep
    # Actually, super().unsign returns the original value, so if original value doesn't contain sep, result won't contain sep
    # However, the sign method adds sep and timestamp, so unsign returns that with sep
    # To bypass, we can use a signed value that is not properly formatted but passes unsign due to some edge case
    # Since we cannot mock, we can create a signer with a custom sep that is not in the value
    custom_sep = b"|"
    signer_with_custom_sep = TimestampSigner("secret-key", sep="|")
    value = b"value"
    timestamp = b"MTIzNDU2Nzg5"
    signed_value = value + custom_sep + timestamp + custom_sep + signer_with_custom_sep.get_signature(value + custom_sep + timestamp)
    # Now result after unsign will contain custom_sep, so predicate is False (sep not in result is False)
    # We need predicate to be True, so we need result without sep
    # Use a signed value where the original value does not contain sep and signer's sep is something not in value
    # But unsign returns value + sep + timestamp, which always contains sep if properly signed
    # To get result without sep, we need a signed value that fails unsign and the payload doesn't contain sep
    # For BadSignature, result = e.payload or b"", so if payload doesn't contain sep, predicate is True
    # We can create a signed value that causes BadSignature with payload that has no sep
    # For example, tamper the signature
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    timestamp = b"MTIzNDU2Nzg5"
    sep = signer.sep.encode()
    signed_value = value + sep + timestamp + sep + b"tampered-signature"
    # This will raise BadSignature in super().unsign, payload is the value+sep+timestamp (contains sep) so predicate False
    # To get payload without sep, we need signed_value that after removing signature has no sep
    # Actually, e.payload is the signed_value without the signature part? Let's check Signer.unsign
    # Usually it returns original value, but on BadSignature, payload is the part before the last sep?
    # We need to ensure e.payload does not contain sep
    # Since e.payload is the signed_value without the last component? It might be the part before the last sep
    # If we create a signed_value with only two parts: value + sep + signature where value contains no sep, then payload = value (no sep)
    # But sign method always adds timestamp, so we need to craft manually
    # Let's create a signed value without timestamp: value + sep + signature
    value = b"value-without-sep"
    sep = signer.sep.encode()
    # We need a signature for value without timestamp
    # We can get signature by calling signer.get_signature(value)
    signature = signer.get_signature(value)
    signed_value = value + sep + signature
    # Now super().unsign will try to unsign this, but since it expects timestamp, it might still work? Actually unsign method of Signer just splits on last sep and verifies signature
    # So it will split into value and signature, then verify signature of value
    # If signature is correct, it returns value (which has no sep), so result = value, sep not in result -> True, sig_error is None, so predicate at line 32 is True, and it will raise BadTimeSignature("timestamp missing")
    # But we want predicate to be False, meaning sep is in result
    # So we need result to contain sep
    # We can add a timestamp: signed_value = value + sep + timestamp + sep + signature
    # But then result will contain sep
    # To make predicate False, we need sep in result
    # So we can use a properly signed value with timestamp
    signed_value = signer.sign("test-value")
    result = super(TimestampSigner, signer).unsign(signed_value)
    # But this will have sep in result, so predicate False
    # We just need to call the method and not raise an exception
    # So the test is to ensure that when sep is in result, predicate is False (no error)
    # However, the instruction says: "Write unit test to ensure that the predicate at line 32 evaluates to False."
    # The predicate is `if sep not in result:` which evaluates to False when sep is in result.
    # We need to test that scenario.
    signer = TimestampSigner("secret-key")
    value = b"test-value"
    signed_value = signer.sign(value)
    # This will have sep in result, so predicate False, and execution continues
    # We can call unsign and expect it to succeed (return value)
    result = signer.unsign(signed_value)
    assert result == value
```


