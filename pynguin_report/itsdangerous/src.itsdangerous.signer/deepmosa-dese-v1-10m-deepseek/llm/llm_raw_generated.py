####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_derive_key_concat():
    signer = Signer("secret-key", salt="salt", key_derivation="concat", digest_method="sha256")
    key = signer.derive_key()
    expected = hashlib.sha256(b"saltsecret-key").digest()
    assert key == expected

def test_derive_key_django_concat():
    signer = Signer("secret-key", salt="salt", key_derivation="django-concat", digest_method="sha256")
    key = signer.derive_key()
    expected = hashlib.sha256(b"salt" + b"signer" + b"secret-key").digest()
    assert key == expected

def test_derive_key_hmac():
    signer = Signer("secret-key", salt="salt", key_derivation="hmac", digest_method="sha256")
    key = signer.derive_key()
    mac = hmac.new(b"secret-key", digestmod="sha256")
    mac.update(b"salt")
    expected = mac.digest()
    assert key == expected

def test_derive_key_none():
    signer = Signer("secret-key", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret-key"

def test_derive_key_with_explicit_secret_key():
    signer = Signer("ignored", key_derivation="none")
    key = signer.derive_key("custom-secret")
    assert key == b"custom-secret"

def test_derive_key_with_bytes_secret_key():
    signer = Signer(b"secret-key", key_derivation="none")
    key = signer.derive_key()
    assert key == b"secret-key"

def test_derive_key_invalid_key_derivation():
    signer = Signer("secret-key", key_derivation="invalid")
    try:
        signer.derive_key()
        assert False
    except TypeError:
        assert True
```


# LLM-generated content at query #2
#--------------------------

def test_unsign_valid_signature():
    signer = Signer("secret-key")
    signed_value = signer.sign("value")
    assert signer.unsign(signed_value) == b"value"

def test_unsign_no_separator():
    signer = Signer("secret-key")
    try:
        signer.unsign(b"noseparator")
        assert False
    except BadSignature:
        pass

def test_unsign_invalid_signature():
    signer = Signer("secret-key")
    try:
        signer.unsign(b"value.invalid")
        assert False
    except BadSignature:
        pass

def test_unsign_with_key_rotation():
    signer = Signer(["old-secret", "new-secret"])
    signed_value = signer.sign("value")
    assert signer.unsign(signed_value) == b"value"

def test_unsign_with_different_salt():
    signer = Signer("secret-key", salt=b"custom-salt")
    signed_value = signer.sign("value")
    assert signer.unsign(signed_value) == b"value"

def test_unsign_with_string_value():
    signer = Signer("secret-key")
    signed_value = signer.sign("value")
    assert signer.unsign(signed_value.decode()) == b"value"

def test_unsign_with_separator_in_value():
    signer = Signer("secret-key", sep=b"|")
    signed_value = signer.sign("value|with|separator")
    assert signer.unsign(signed_value) == b"value|with|separator"


# LLM-generated content at query #3
#--------------------------

def test_signer_constructor_default_secret_key_str():
    signer = Signer("my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_secret_key_bytes():
    signer = Signer(b"my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]

def test_signer_constructor_secret_key_list():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_custom_sep():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_custom_salt():
    signer = Signer("secret", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_custom_digest_method():
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_custom_algorithm():
    class MockAlgorithm(SigningAlgorithm):
        def get_signature(self, key, value):
            return b"sig"
    algo = MockAlgorithm()
    signer = Signer("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_signer_constructor_salt_none():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_sep_in_base64_alphabet():
    import string
    from itsdangerous.exc import BadSignature
    for ch in string.ascii_letters + string.digits + "-_=":
        try:
            Signer("secret", sep=ch)
            assert False
        except ValueError:
            pass


# LLM-generated content at query #4
#--------------------------

def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"test value"
    assert signer.verify_signature(value, b"invalid_sig") == False

def test_verify_signature_with_empty_value():
    signer = Signer("secret-key")
    sig = signer.get_signature(b"")
    assert signer.verify_signature(b"", sig) == True

def test_verify_signature_with_bytes_value():
    signer = Signer("secret-key")
    value = b"bytes value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_string_value():
    signer = Signer("secret-key")
    value = "string value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_different_secret_key():
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    different_signer = Signer("different-secret")
    assert different_signer.verify_signature(value, sig) == False

def test_verify_signature_with_key_rotation():
    signer = Signer(["old-key", "new-key"])
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_invalid_base64_sig():
    signer = Signer("secret-key")
    value = b"test value"
    assert signer.verify_signature(value, b"!!!invalid_base64!!!") == False

def test_verify_signature_with_non_bytes_sig():
    signer = Signer("secret-key")
    value = b"test value"
    assert signer.verify_signature(value, "not bytes") == False


# LLM-generated content at query #5
#--------------------------

def test_verify_signature_with_valid_signature_returns_true():
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_invalid_signature_returns_false():
    signer = Signer("secret-key")
    value = b"test value"
    invalid_sig = b"invalid_signature"
    assert signer.verify_signature(value, invalid_sig) is False

def test_verify_signature_with_wrong_key_returns_false():
    signer1 = Signer("key1")
    signer2 = Signer("key2")
    value = b"test value"
    sig = signer1.get_signature(value)
    assert signer2.verify_signature(value, sig) is False

def test_verify_signature_with_key_rotation_returns_true():
    signer = Signer(["old_key", "new_key"])
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_empty_value_returns_true():
    signer = Signer("secret-key")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_string_value_returns_true():
    signer = Signer("secret-key")
    value = "test string"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_base64_decode_exception_returns_false():
    signer = Signer("secret-key")
    value = b"test value"
    invalid_sig = b"!!!invalid_base64!!!"
    assert signer.verify_signature(value, invalid_sig) is False

def test_verify_signature_with_none_sig_returns_false():
    signer = Signer("secret-key")
    value = b"test value"
    assert signer.verify_signature(value, None) is False


# LLM-generated content at query #6
#--------------------------

```python
def test_signer_constructor_defaults():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_signer_constructor_with_list_of_strings():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_list_of_bytes():
    signer = Signer([b"key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_key_derivation_concat():
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_key_derivation_hmac():
    signer = Signer("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_key_derivation_none():
    signer = Signer("secret-key", key_derivation="none")
    assert signer.key_derivation == "none"

def test_signer_constructor_with_digest_method():
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_with_invalid_sep():
    import re
    try:
        Signer("secret-key", sep=b".")
    except ValueError as e:
        assert "separator" in str(e).lower()
```


# LLM-generated content at query #7
#--------------------------

def test_verify_signature_exception_not_raised_for_valid_base64_input():
    signer = Signer(secret_key="test-secret", salt="test-salt")
    value = b"test-value"
    sig = signer.get_signature(value)
    result = signer.verify_signature(value, sig)
    assert result == True


# LLM-generated content at query #8
#--------------------------

```python
def test_verify_signature_predicate_false():
    signer = Signer("test-secret-key")
    result = signer.verify_signature(b"test-value", b"invalid-base64!!!")
    assert result == False
```


# LLM-generated content at query #9
#--------------------------

def test_signer_init_with_string_secret_key():
    signer = Signer("my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_init_with_bytes_secret_key():
    signer = Signer(b"my-secret-key")
    assert signer.secret_keys == [b"my-secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_init_with_list_of_strings():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_init_with_list_of_bytes():
    signer = Signer([b"key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_init_with_custom_sep():
    signer = Signer("key", sep=b":")
    assert signer.sep == b":"

def test_signer_init_with_custom_salt():
    signer = Signer("key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_init_with_custom_key_derivation():
    signer = Signer("key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_init_with_custom_digest_method():
    signer = Signer("key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_init_with_custom_algorithm():
    algo = HMACAlgorithm(hashlib.sha256)
    signer = Signer("key", algorithm=algo)
    assert signer.algorithm == algo

def test_signer_init_with_sep_in_base64_alphabet_raises():
    import pytest
    with pytest.raises(ValueError):
        Signer("key", sep=b"a")

def test_signer_init_with_none_salt():
    signer = Signer("key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #10
#--------------------------

def test_signer_constructor_defaults() -> None:
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.digest_method, type)
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret() -> None:
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_key_list() -> None:
    signer = Signer(["old_key", b"new_key"])
    assert signer.secret_keys == [b"old_key", b"new_key"]

def test_signer_constructor_with_salt() -> None:
    signer = Signer("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_sep() -> None:
    signer = Signer("secret", sep=":")
    assert signer.sep == b":"

def test_signer_constructor_with_key_derivation() -> None:
    signer = Signer("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_digest_method() -> None:
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_algorithm() -> None:
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_sep_in_base64_alphabet_raises() -> None:
    import pytest
    with pytest.raises(ValueError):
        Signer("secret", sep="a")


# LLM-generated content at query #11
#--------------------------

def test_verify_signature_valid_signature():
    signer = Signer("secret-key")
    value = b"test value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_invalid_signature():
    signer = Signer("secret-key")
    value = b"test value"
    signature = b"invalid_signature"
    assert signer.verify_signature(value, signature) == False

def test_verify_signature_empty_value():
    signer = Signer("secret-key")
    value = b""
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_none_value():
    signer = Signer("secret-key")
    value = None
    signature = b"some_signature"
    assert signer.verify_signature(value, signature) == False

def test_verify_signature_string_value():
    signer = Signer("secret-key")
    value = "test value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_string_signature():
    signer = Signer("secret-key")
    value = b"test value"
    signature = signer.get_signature(value).decode("ascii")
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_key_rotation():
    signer = Signer(["old_key", "new_key"])
    value = b"test value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_key_rotation_old_key():
    signer = Signer(["old_key", "new_key"])
    value = b"test value"
    old_signer = Signer("old_key")
    signature = old_signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_base64_decode_failure():
    signer = Signer("secret-key")
    value = b"test value"
    signature = b"!!!invalid_base64!!!"
    assert signer.verify_signature(value, signature) == False


# LLM-generated content at query #12
#--------------------------

def test_signer_constructor_default_parameters() -> None:
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.sep == b"."
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key() -> None:
    signer = Signer(b"secret-bytes")
    assert signer.secret_keys == [b"secret-bytes"]

def test_signer_constructor_with_secret_key_list() -> None:
    signer = Signer(["key1", "key2", b"key3"])
    assert signer.secret_keys == [b"key1", b"key2", b"key3"]

def test_signer_constructor_with_custom_salt() -> None:
    signer = Signer("secret", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_custom_separator() -> None:
    signer = Signer("secret", sep=":")
    assert signer.sep == b":"

def test_signer_constructor_with_invalid_separator() -> None:
    try:
        Signer("secret", sep="a")
        assert False
    except ValueError:
        pass

def test_signer_constructor_with_custom_key_derivation() -> None:
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method() -> None:
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm() -> None:
    class MockAlgorithm(SigningAlgorithm):
        def get_signature(self, key: bytes, value: bytes) -> bytes:
            return b"mock"
    algo = MockAlgorithm()
    signer = Signer("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_signer_constructor_salt_none() -> None:
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_key_derivation_none() -> None:
    signer = Signer("secret", key_derivation=None)
    assert signer.key_derivation == "django-concat"

def test_signer_constructor_digest_method_none() -> None:
    signer = Signer("secret", digest_method=None)
    assert signer.digest_method is not None

def test_signer_constructor_algorithm_none() -> None:
    signer = Signer("secret", algorithm=None)
    assert isinstance(signer.algorithm, HMACAlgorithm)


# LLM-generated content at query #13
#--------------------------

def test_verify_signature_invalid_base64_returns_false():
    signer = Signer("secret-key")
    result = signer.verify_signature(b"test", b"!!!invalid-base64!!!")
    assert result is False


# LLM-generated content at query #14
#--------------------------

def test_signer_constructor_default_parameters():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_signer_constructor_with_list_secret_keys():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_sep_string():
    signer = Signer("secret-key", sep=":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_custom_salt_string():
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_none_salt():
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method():
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    custom_algorithm = HMACAlgorithm(hashlib.sha256)
    signer = Signer("secret-key", algorithm=custom_algorithm)
    assert signer.algorithm == custom_algorithm

def test_signer_constructor_with_sep_in_base64_alphabet():
    import string
    for char in string.ascii_letters + string.digits + "-_=":
        try:
            Signer("secret-key", sep=char)
            assert False, f"Should raise ValueError for sep={char!r}"
        except ValueError:
            pass


# LLM-generated content at query #15
#--------------------------

def test_verify_signature_raises_exception_and_returns_false():
    signer = Signer(secret_key="test-secret")
    value = b"test-value"
    sig = "invalid-base64!!!"
    result = signer.verify_signature(value, sig)
    assert result == False


# LLM-generated content at query #16
#--------------------------

def test_signer_constructor_default_parameters() -> None:
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.algorithm.digest_method == hashlib.sha1

def test_signer_constructor_with_bytes_secret_key() -> None:
    signer = Signer(b"bytes-secret")
    assert signer.secret_keys == [b"bytes-secret"]

def test_signer_constructor_with_list_of_secret_keys() -> None:
    signer = Signer(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_signer_constructor_with_custom_sep() -> None:
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_sep_in_base64_alphabet_raises() -> None:
    try:
        Signer("secret", sep=b"a")
        assert False
    except ValueError:
        pass

def test_signer_constructor_with_none_salt() -> None:
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_salt() -> None:
    signer = Signer("secret", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_custom_key_derivation() -> None:
    signer = Signer("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_custom_digest_method() -> None:
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm() -> None:
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = Signer("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm
    assert signer.algorithm.digest_method == hashlib.sha256


# LLM-generated content at query #17
#--------------------------

def test_signer_constructor_defaults() -> None:
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.digest_method(), type(hashlib.sha1()))
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes() -> None:
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_signer_constructor_with_key_list() -> None:
    signer = Signer(["key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_sep() -> None:
    signer = Signer("secret-key", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt() -> None:
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_none_salt() -> None:
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_key_derivation() -> None:
    signer = Signer("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_digest_method() -> None:
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_signer_constructor_with_algorithm() -> None:
    class MockAlgorithm(SigningAlgorithm):
        def get_signature(self, key: bytes, value: bytes) -> bytes:
            return b"mock"
    algorithm = MockAlgorithm()
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_sep_in_base64_alphabet_raises() -> None:
    import pytest
    with pytest.raises(ValueError):
        Signer("secret-key", sep=b"=")


# LLM-generated content at query #18
#--------------------------

def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"test"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"test"
    sig = b"invalid_signature"
    assert signer.verify_signature(value, sig) == False

def test_verify_signature_with_text_value_and_bytes_sig():
    signer = Signer("secret-key")
    value = "test"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_bytes_value_and_text_sig():
    signer = Signer("secret-key")
    value = b"test"
    sig = signer.get_signature(value).decode()
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_multiple_secret_keys_oldest():
    signer = Signer(["old_key", "new_key"])
    value = b"test"
    sig = signer.get_signature(value)
    signer.secret_keys = [b"old_key", b"new_key"]
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_multiple_secret_keys_only_oldest_valid():
    signer = Signer(["old_key", "new_key"])
    value = b"test"
    sig = signer.get_signature(value)
    signer.secret_keys = [b"old_key"]
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_multiple_secret_keys_only_newest_valid():
    signer = Signer(["old_key", "new_key"])
    value = b"test"
    sig = signer.get_signature(value)
    signer.secret_keys = [b"new_key"]
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_none_secret_key():
    signer = Signer("secret-key")
    value = b"test"
    sig = signer.get_signature(value)
    signer.secret_keys = [b"different_key"]
    assert signer.verify_signature(value, sig) == False

def test_verify_signature_with_empty_value():
    signer = Signer("secret-key")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_base64_decode_failure():
    signer = Signer("secret-key")
    value = b"test"
    sig = b"!!!invalid_base64!!!"
    assert signer.verify_signature(value, sig) == False


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    invalid_sig = b"invalid_signature"
    assert signer.verify_signature(value, invalid_sig) == False

def test_verify_signature_with_empty_value():
    signer = Signer("secret-key", salt="salt")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_different_secret_key():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    sig = signer.get_signature(value)
    different_signer = Signer("different-secret-key", salt="salt")
    assert different_signer.verify_signature(value, sig) == False

def test_verify_signature_with_different_salt():
    signer = Signer("secret-key", salt="salt1")
    value = b"test value"
    sig = signer.get_signature(value)
    different_signer = Signer("secret-key", salt="salt2")
    assert different_signer.verify_signature(value, sig) == False

def test_verify_signature_with_invalid_base64_signature():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    invalid_sig = b"!!!invalid_base64!!!"
    assert signer.verify_signature(value, invalid_sig) == False

def test_verify_signature_with_none_signature():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    assert signer.verify_signature(value, None) == False

def test_verify_signature_with_bytes_value():
    signer = Signer("secret-key", salt="salt")
    value = b"bytes value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_string_value():
    signer = Signer("secret-key", salt="salt")
    value = "string value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_key_rotation():
    signer = Signer(["old-key", "new-key"], salt="salt")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_old_key_rotation():
    signer = Signer(["old-key", "new-key"], salt="salt")
    value = b"test value"
    sig = signer.get_signature(value)
    old_signer = Signer("old-key", salt="salt")
    old_sig = old_signer.get_signature(value)
    assert signer.verify_signature(value, old_sig) == True

def test_verify_signature_with_none_secret_key():
    signer = Signer(None, salt="salt")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True


# LLM-generated content at query #2
#--------------------------

```python
def test_derive_key_with_none_secret_key_uses_last_secret_key():
    signer = Signer(secret_key=b"secret")
    result = signer.derive_key()
    assert isinstance(result, bytes)

def test_derive_key_with_custom_secret_key():
    signer = Signer(secret_key=b"secret")
    result = signer.derive_key(secret_key=b"custom")
    assert isinstance(result, bytes)

def test_derive_key_with_concat_method():
    signer = Signer(secret_key=b"secret", key_derivation="concat")
    result = signer.derive_key()
    assert isinstance(result, bytes)

def test_derive_key_with_django_concat_method():
    signer = Signer(secret_key=b"secret", key_derivation="django-concat")
    result = signer.derive_key()
    assert isinstance(result, bytes)

def test_derive_key_with_hmac_method():
    signer = Signer(secret_key=b"secret", key_derivation="hmac")
    result = signer.derive_key()
    assert isinstance(result, bytes)

def test_derive_key_with_none_method():
    signer = Signer(secret_key=b"secret", key_derivation="none")
    result = signer.derive_key()
    assert result == b"secret"

def test_derive_key_raises_type_error_on_unknown_method():
    signer = Signer(secret_key=b"secret", key_derivation="unknown")
    try:
        signer.derive_key()
        assert False
    except TypeError:
        pass

def test_derive_key_with_string_secret_key():
    signer = Signer(secret_key="secret")
    result = signer.derive_key()
    assert isinstance(result, bytes)

def test_derive_key_with_custom_string_secret_key():
    signer = Signer(secret_key=b"secret")
    result = signer.derive_key(secret_key="custom")
    assert isinstance(result, bytes)
```


# LLM-generated content at query #3
#--------------------------

def test_verify_signature_with_valid_signature_returns_true():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_invalid_signature_returns_false():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    sig = b"invalid_signature"
    assert signer.verify_signature(value, sig) is False

def test_verify_signature_with_malformed_base64_signature_returns_false():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    sig = b"!!!not_base64!!!"
    assert signer.verify_signature(value, sig) is False

def test_verify_signature_with_empty_value_returns_true_for_corresponding_signature():
    signer = Signer("secret-key", salt="salt")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_different_secret_key_returns_false():
    signer1 = Signer("secret-key-1", salt="salt")
    signer2 = Signer("secret-key-2", salt="salt")
    value = b"test value"
    sig = signer1.get_signature(value)
    assert signer2.verify_signature(value, sig) is False

def test_verify_signature_with_multiple_secret_keys_returns_true_for_any_key():
    signer = Signer(["old-key", "new-key"], salt="salt")
    value = b"test value"
    sig = signer.get_signature(value)  # signed with "new-key"
    # Create a signer with "old-key" only to verify old key works
    signer_old = Signer("old-key", salt="salt")
    sig_old = signer_old.get_signature(value)
    assert signer.verify_signature(value, sig_old) is True

def test_verify_signature_with_bytes_value_and_str_signature():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    sig = signer.get_signature(value).decode("ascii")
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_str_value_and_bytes_signature():
    signer = Signer("secret-key", salt="salt")
    value = "test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_different_salt_returns_false():
    signer1 = Signer("secret-key", salt="salt1")
    signer2 = Signer("secret-key", salt="salt2")
    value = b"test value"
    sig = signer1.get_signature(value)
    assert signer2.verify_signature(value, sig) is False

def test_verify_signature_with_none_algorithm_uses_default_hmac():
    signer = Signer("secret-key", salt="salt")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True


# LLM-generated content at query #4
#--------------------------

def test_signer_constructor_defaults():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.digest_method == hashlib.sha1

def test_signer_constructor_with_bytes_secret():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_list_of_strings():
    signer = Signer(["old", "new"])
    assert signer.secret_keys == [b"old", b"new"]

def test_signer_constructor_with_list_of_bytes():
    signer = Signer([b"old", b"new"])
    assert signer.secret_keys == [b"old", b"new"]

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt=b"custom")
    assert signer.salt == b"custom"

def test_signer_constructor_with_string_salt():
    signer = Signer("secret", salt="custom")
    assert signer.salt == b"custom"

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_string_sep():
    signer = Signer("secret", sep=":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method():
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algo = HMACAlgorithm(hashlib.sha256)
    signer = Signer("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_signer_constructor_sep_raises_on_base64_char():
    import base64
    try:
        Signer("secret", sep=b"A")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_signer_constructor_with_none_salt():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #5
#--------------------------

def test_signer_constructor_defaults():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_salt():
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_sep():
    signer = Signer("secret-key", sep="|")
    assert signer.sep == b"|"

def test_signer_constructor_with_key_derivation():
    signer = Signer("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_digest_method():
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_with_secret_key_list():
    signer = Signer(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_signer_constructor_with_secret_key_bytes():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_signer_constructor_with_secret_key_iterable():
    signer = Signer(iter(["key1", "key2"]))
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_sep_in_base64_alphabet():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret-key", sep="-")

def test_signer_constructor_with_none_salt():
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #6
#--------------------------

def test_signer_constructor_defaults() -> None:
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_key() -> None:
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_signer_constructor_with_list_of_keys() -> None:
    signer = Signer(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_signer_constructor_with_list_of_bytes_keys() -> None:
    signer = Signer([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_signer_constructor_with_custom_sep() -> None:
    signer = Signer("secret-key", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt() -> None:
    signer = Signer("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_custom_key_derivation() -> None:
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method() -> None:
    from hashlib import sha256
    signer = Signer("secret-key", digest_method=sha256)
    assert signer.digest_method is sha256

def test_signer_constructor_with_custom_algorithm() -> None:
    algorithm = HMACAlgorithm()
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_with_sep_in_base64_alphabet() -> None:
    try:
        Signer("secret-key", sep=b"a")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_signer_constructor_with_none_salt() -> None:
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #7
#--------------------------

```
def test_verify_signature_predicate_false():
    signer = Signer(secret_key="test", salt=None)
    result = signer.verify_signature(value=b"test_value", sig=b"invalid_base64!!!")
    assert result is False
```


# LLM-generated content at query #8
#--------------------------

def test_signer_constructor_default_parameters():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.algorithm.digest_method is signer.digest_method

def test_signer_constructor_with_bytes_secret():
    signer = Signer(b"secret_bytes")
    assert signer.secret_keys == [b"secret_bytes"]

def test_signer_constructor_with_secret_key_list():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_custom_sep():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_custom_salt():
    signer = Signer("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_salt_none():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_signer_constructor_custom_algorithm():
    algo = HMACAlgorithm()
    signer = Signer("secret", algorithm=algo)
    assert signer.algorithm is algo

def test_signer_constructor_sep_in_base64_alphabet():
    try:
        Signer("secret", sep=b".")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for sep in base64 alphabet"


# LLM-generated content at query #9
#--------------------------

def test_verify_signature_with_valid_signature_returns_true():
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_invalid_signature_returns_false():
    signer = Signer("secret-key")
    value = b"test value"
    sig = b"invalid_signature"
    assert signer.verify_signature(value, sig) is False

def test_verify_signature_with_empty_string_returns_false():
    signer = Signer("secret-key")
    value = b"test value"
    sig = b""
    assert signer.verify_signature(value, sig) is False

def test_verify_signature_with_non_base64_signature_returns_false():
    signer = Signer("secret-key")
    value = b"test value"
    sig = b"!!!invalid_base64!!!"
    assert signer.verify_signature(value, sig) is False

def test_verify_signature_with_string_value_returns_true():
    signer = Signer("secret-key")
    value = "test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_different_secret_key_returns_false():
    signer1 = Signer("secret-key-1")
    signer2 = Signer("secret-key-2")
    value = b"test value"
    sig = signer1.get_signature(value)
    assert signer2.verify_signature(value, sig) is False

def test_verify_signature_with_multiple_secret_keys_returns_true():
    signer = Signer(["old-key", "new-key"])
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True


# LLM-generated content at query #10
#--------------------------

def test_signer_constructor_defaults() -> None:
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret() -> None:
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_signer_constructor_with_list_secret() -> None:
    signer = Signer(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_signer_constructor_with_custom_sep() -> None:
    signer = Signer("secret-key", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt() -> None:
    signer = Signer("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_none_salt() -> None:
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_key_derivation() -> None:
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_digest_method() -> None:
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm() -> None:
    algorithm = HMACAlgorithm()
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_invalid_sep_raises() -> None:
    import re
    try:
        Signer("secret-key", sep=b"-")
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "separator" in str(e).lower()


# LLM-generated content at query #11
#--------------------------

def test_signer_constructor_defaults():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_list_of_strings():
    signer = Signer(["old", "new"])
    assert signer.secret_keys == [b"old", b"new"]

def test_signer_constructor_with_list_of_bytes():
    signer = Signer([b"old", b"new"])
    assert signer.secret_keys == [b"old", b"new"]

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_salt_none():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_sep_in_base64_alphabet_raises():
    import pytest
    try:
        Signer("secret", sep=b"a")
        assert False
    except ValueError:
        pass


# LLM-generated content at query #12
#--------------------------

def test_verify_signature_returns_false_when_base64_decode_raises_exception():
    signer = Signer("secret-key")
    result = signer.verify_signature(b"test_value", b"!!!invalid_base64!!!")
    assert result == False


# LLM-generated content at query #13
#--------------------------

def test_verify_signature_with_valid_signature_returns_true():
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_invalid_signature_returns_false():
    signer = Signer("secret-key")
    value = b"test value"
    invalid_sig = b"invalid_signature"
    assert signer.verify_signature(value, invalid_sig) is False

def test_verify_signature_with_empty_value_returns_true():
    signer = Signer("secret-key")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_bytes_input_returns_true():
    signer = Signer("secret-key")
    value = b"bytes value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_string_input_returns_true():
    signer = Signer("secret-key")
    value = "string value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_base64_decode_failure_returns_false():
    signer = Signer("secret-key")
    value = b"test value"
    invalid_base64_sig = b"!!!invalid base64"
    assert signer.verify_signature(value, invalid_base64_sig) is False

def test_verify_signature_with_multiple_secret_keys_returns_true():
    signer = Signer(["old-key", "new-key"])
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) is True

def test_verify_signature_with_multiple_secret_keys_and_old_sig_returns_true():
    old_signer = Signer("old-key")
    value = b"test value"
    old_sig = old_signer.get_signature(value)
    new_signer = Signer(["old-key", "new-key"])
    assert new_signer.verify_signature(value, old_sig) is True

def test_verify_signature_with_wrong_secret_key_returns_false():
    signer1 = Signer("key1")
    signer2 = Signer("key2")
    value = b"test value"
    sig = signer1.get_signature(value)
    assert signer2.verify_signature(value, sig) is False


# LLM-generated content at query #14
#--------------------------

def test_signer_constructor_default_parameters():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.algorithm.digest_method == signer.default_digest_method

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_list_of_secret_keys():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]
    assert signer.secret_key == b"key2"
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_separator():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_salt_none():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256
    assert signer.algorithm.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_with_separator_in_base64_alphabet_raises_error():
    import re
    from itsdangerous.exc import BadSignature
    try:
        Signer("secret", sep=b"+")
    except ValueError as e:
        assert "The given separator cannot be used" in str(e)


# LLM-generated content at query #15
#--------------------------

def test_signer_constructor_default_separator():
    signer = Signer("secret-key", salt=b"my-salt")
    assert signer.sep == b"."

def test_signer_constructor_custom_separator():
    signer = Signer("secret-key", salt=b"my-salt", sep=b"|")
    assert signer.sep == b"|"

def test_signer_constructor_separator_invalid():
    import re
    try:
        Signer("secret-key", salt=b"my-salt", sep=b"a")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_signer_constructor_salt_none():
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_salt_bytes():
    signer = Signer("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_salt_str():
    signer = Signer("secret-key", salt="custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_key_derivation_default():
    signer = Signer("secret-key")
    assert signer.key_derivation == "django-concat"

def test_signer_constructor_key_derivation_custom():
    signer = Signer("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_digest_method_default():
    signer = Signer("secret-key")
    import hashlib
    assert signer.digest_method == hashlib.sha1

def test_signer_constructor_digest_method_custom():
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_algorithm_default():
    signer = Signer("secret-key")
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_algorithm_custom():
    class MockAlgorithm(SigningAlgorithm):
        def get_signature(self, key, value):
            return b"mock"
    algorithm = MockAlgorithm()
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_secret_keys_single_string():
    signer = Signer("my-secret")
    assert signer.secret_keys == [b"my-secret"]

def test_signer_constructor_secret_keys_single_bytes():
    signer = Signer(b"my-secret")
    assert signer.secret_keys == [b"my-secret"]

def test_signer_constructor_secret_keys_list():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_secret_keys_list_bytes():
    signer = Signer([b"key1", b"key2"])
    assert signer.secret_keys == [b"key1", b"key2"]


# LLM-generated content at query #16
#--------------------------

def test_verify_signature_invalid_base64_returns_false():
    signer = Signer(secret_key="test-secret", salt="test-salt")
    result = signer.verify_signature(b"test-value", "invalid-base64!!!")
    assert result is False


# LLM-generated content at query #17
#--------------------------

def test_verify_signature_base64_decode_false():
    signer = Signer(secret_key="secret")
    # Provide an invalid base64 string that will cause base64_decode to raise an exception
    result = signer.verify_signature(b"value", b"!!!invalid-base64!!!")
    assert not result


# LLM-generated content at query #18
#--------------------------

def test_signer_constructor_default_parameters():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method == hashlib.sha1
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_list_of_secrets():
    signer = Signer(["old_secret", "new_secret"])
    assert signer.secret_keys == [b"old_secret", b"new_secret"]

def test_signer_constructor_with_list_of_bytes_secrets():
    signer = Signer([b"old_secret", b"new_secret"])
    assert signer.secret_keys == [b"old_secret", b"new_secret"]

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_sep_string():
    signer = Signer("secret", sep=":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt=b"custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_custom_salt_string():
    signer = Signer("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_salt_none():
    signer = Signer("secret", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_custom_digest_method():
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm(hashlib.sha256)
    signer = Signer("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_sep_in_base64_alphabet_raises():
    import string
    for sep in string.ascii_letters + string.digits + "-_=":
        try:
            Signer("secret", sep=sep)
            assert False, f"Expected ValueError for sep {sep!r}"
        except ValueError:
            pass

def test_signer_constructor_sep_bytes_in_base64_alphabet_raises():
    for sep in [b"a", b"Z", b"0", b"-", b"_", b"="]:
        try:
            Signer("secret", sep=sep)
            assert False, f"Expected ValueError for sep {sep!r}"
        except ValueError:
            pass


