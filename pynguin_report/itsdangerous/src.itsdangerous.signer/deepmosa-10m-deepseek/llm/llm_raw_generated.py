####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_derive_key_default_secret_key():
    signer = Signer("secret")
    key = signer.derive_key()
    assert isinstance(key, bytes)
    assert len(key) > 0

def test_derive_key_with_specific_secret():
    signer = Signer("secret")
    key = signer.derive_key("other-secret")
    assert isinstance(key, bytes)

def test_derive_key_with_bytes_secret():
    signer = Signer(b"secret")
    key = signer.derive_key(b"other-secret")
    assert isinstance(key, bytes)

def test_derive_key_concat():
    signer = Signer("secret", key_derivation="concat")
    key = signer.derive_key()
    assert isinstance(key, bytes)

def test_derive_key_django_concat():
    signer = Signer("secret", key_derivation="django-concat")
    key = signer.derive_key()
    assert isinstance(key, bytes)

def test_derive_key_hmac():
    signer = Signer("secret", key_derivation="hmac")
    key = signer.derive_key()
    assert isinstance(key, bytes)

def test_derive_key_none():
    signer = Signer("secret", key_derivation="none")
    key = signer.derive_key("test-key")
    assert key == b"test-key"

def test_derive_key_unknown_derivation():
    signer = Signer("secret", key_derivation="unknown")
    try:
        signer.derive_key()
        assert False
    except TypeError:
        pass


# LLM-generated content at query #2
#--------------------------

def test_unsign_valid_signature():
    signer = Signer("secret-key")
    signed = signer.sign("test")
    result = signer.unsign(signed)
    assert result == b"test"

def test_unsign_invalid_separator():
    signer = Signer("secret-key")
    try:
        signer.unsign(b"novalue")
        assert False
    except BadSignature:
        pass

def test_unsign_wrong_signature():
    signer = Signer("secret-key")
    signed = signer.sign("test")
    tampered = signed[:-1] + b"x"
    try:
        signer.unsign(tampered)
        assert False
    except BadSignature:
        pass

def test_unsign_with_multiple_separators():
    signer = Signer("secret-key")
    signed = signer.sign("test.value")
    result = signer.unsign(signed)
    assert result == b"test.value"

def test_unsign_with_key_rotation():
    signer = Signer(["old-key", "new-key"])
    signed_with_new = signer.sign("test")
    result = signer.unsign(signed_with_new)
    assert result == b"test"
    signer_with_old_only = Signer("old-key")
    old_signed = signer_with_old_only.sign("test")
    result = signer.unsign(old_signed)
    assert result == b"test"


# LLM-generated content at query #3
#--------------------------

def test_signer_constructor_defaults():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_signer_constructor_with_list_of_strings():
    signer = Signer(["old-key", "new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_signer_constructor_with_list_of_bytes():
    signer = Signer([b"old-key", b"new-key"])
    assert signer.secret_keys == [b"old-key", b"new-key"]

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_none_salt():
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_separator_in_base64_alphabet_raises():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret-key", sep=b"a")


# LLM-generated content at query #4
#--------------------------

def test_verify_signature_valid_signature():
    signer = Signer(secret_key="secret-key")
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_invalid_signature():
    signer = Signer(secret_key="secret-key")
    value = b"test_value"
    sig = b"invalid_signature"
    assert signer.verify_signature(value, sig) == False

def test_verify_signature_empty_value():
    signer = Signer(secret_key="secret-key")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_none_value():
    signer = Signer(secret_key="secret-key")
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_multiple_keys():
    signer = Signer(secret_key=["old_key", "new_key"])
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_old_key():
    signer = Signer(secret_key=["old_key", "new_key"])
    value = b"test_value"
    old_signer = Signer(secret_key="old_key")
    sig = old_signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_wrong_secret():
    signer1 = Signer(secret_key="secret1")
    signer2 = Signer(secret_key="secret2")
    value = b"test_value"
    sig = signer1.get_signature(value)
    assert signer2.verify_signature(value, sig) == False

def test_verify_signature_base64_decode_exception():
    signer = Signer(secret_key="secret-key")
    value = b"test_value"
    sig = b"!!!invalid_base64!!!"
    assert signer.verify_signature(value, sig) == False

def test_verify_signature_invalid_value_type():
    signer = Signer(secret_key="secret-key")
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(123, sig) == False

def test_verify_signature_invalid_sig_type():
    signer = Signer(secret_key="secret-key")
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, 456) == False

def test_verify_signature_different_salt():
    signer1 = Signer(secret_key="secret-key", salt=b"salt1")
    signer2 = Signer(secret_key="secret-key", salt=b"salt2")
    value = b"test_value"
    sig = signer1.get_signature(value)
    assert signer2.verify_signature(value, sig) == False

def test_verify_signature_different_separator():
    signer = Signer(secret_key="secret-key", sep=b"|")
    value = b"test_value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True


# LLM-generated content at query #5
#--------------------------

def test_verify_signature_returns_false_when_base64_decode_raises_exception():
    signer = Signer(secret_key="secret")
    value = b"test"
    sig = "invalid base64!!!"
    result = signer.verify_signature(value, sig)
    assert result == False


# LLM-generated content at query #6
#--------------------------

```python
def test_signer_constructor_defaults():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is Signer.default_digest_method
    assert isinstance(signer.algorithm, HMACAlgorithm)
    assert signer.algorithm.digest_method is Signer.default_digest_method

def test_signer_constructor_with_bytes_secret():
    signer = Signer(b"secret-key")
    assert signer.secret_keys == [b"secret-key"]

def test_signer_constructor_with_list_secret():
    signer = Signer(["key1", "key2", "key3"])
    assert signer.secret_keys == [b"key1", b"key2", b"key3"]

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_custom_sep():
    signer = Signer("secret-key", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret-key", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256
    assert signer.algorithm.digest_method is hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algo = HMACAlgorithm()
    signer = Signer("secret-key", algorithm=algo)
    assert signer.algorithm is algo

def test_signer_constructor_sep_in_base64_alphabet_raises_error():
    import pytest
    for sep in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_=":
        with pytest.raises(ValueError):
            Signer("secret-key", sep=sep)

def test_signer_constructor_salt_none_uses_default():
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"

def test_signer_constructor_empty_list_secret():
    signer = Signer([])
    assert signer.secret_keys == []

def test_signer_constructor_secret_key_property():
    signer = Signer(["old", "new"])
    assert signer.secret_key == b"new"
```


# LLM-generated content at query #7
#--------------------------

def test_verify_signature_returns_false_on_base64_decode_exception():
    signer = Signer(secret_key="test-secret", salt="test-salt")
    result = signer.verify_signature(b"test-value", b"!!!invalid-base64!!!")
    assert result is False


# LLM-generated content at query #8
#--------------------------

def test_verify_signature_returns_false_when_base64_decode_raises_exception():
    signer = Signer(secret_key=b"test-secret")
    result = signer.verify_signature(b"test-value", b"invalid-base64!!!")
    assert result == False


# LLM-generated content at query #9
#--------------------------

def test_verify_signature_valid_signature():
    signer = Signer("secret-key")
    value = b"test"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_invalid_signature():
    signer = Signer("secret-key")
    value = b"test"
    invalid_sig = b"invalid_signature"
    assert signer.verify_signature(value, invalid_sig) == False

def test_verify_signature_empty_string():
    signer = Signer("secret-key")
    assert signer.verify_signature(b"", b"") == False

def test_verify_signature_with_different_secret_key():
    signer1 = Signer("secret1")
    signer2 = Signer("secret2")
    value = b"test"
    signature = signer1.get_signature(value)
    assert signer2.verify_signature(value, signature) == False

def test_verify_signature_with_key_rotation_valid():
    signer = Signer(["old_secret", "new_secret"])
    value = b"test"
    old_sig = Signer("old_secret").get_signature(value)
    assert signer.verify_signature(value, old_sig) == True

def test_verify_signature_with_key_rotation_invalid():
    signer = Signer(["old_secret", "new_secret"])
    value = b"test"
    wrong_sig = Signer("wrong_secret").get_signature(value)
    assert signer.verify_signature(value, wrong_sig) == False

def test_verify_signature_with_unicode_string():
    signer = Signer("secret-key")
    value = "héllo"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_custom_separator():
    signer = Signer("secret-key", sep=b"|")
    value = b"test"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_none_salt():
    signer = Signer("secret-key", salt=None)
    value = b"test"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_different_encoding():
    signer = Signer("secret-key")
    value = b"\xff\xfe"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True


# LLM-generated content at query #10
#--------------------------

def test_verify_signature_valid_signature():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_invalid_signature():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b"test_value"
    invalid_sig = b"invalidsignature"
    assert signer.verify_signature(value, invalid_sig) == False

def test_verify_signature_wrong_key():
    signer1 = Signer("secret-key-1", salt="salt", sep=".")
    signer2 = Signer("secret-key-2", salt="salt", sep=".")
    value = b"test_value"
    signature = signer1.get_signature(value)
    assert signer2.verify_signature(value, signature) == False

def test_verify_signature_empty_value():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b""
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_empty_sig():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b"test_value"
    assert signer.verify_signature(value, b"") == False

def test_verify_signature_unicode_string_value():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = "test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_unicode_string_sig():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b"test_value"
    signature = signer.get_signature(value).decode("ascii")
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_invalid_base64_sig():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b"test_value"
    invalid_base64_sig = b"!!!invalidbase64!!!"
    assert signer.verify_signature(value, invalid_base64_sig) == False

def test_verify_signature_with_key_rotation():
    signer = Signer(["old-secret", "new-secret"], salt="salt", sep=".")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_old_key():
    old_signer = Signer("old-secret", salt="salt", sep=".")
    new_signer = Signer(["old-secret", "new-secret"], salt="salt", sep=".")
    value = b"test_value"
    old_signature = old_signer.get_signature(value)
    assert new_signer.verify_signature(value, old_signature) == True

def test_verify_signature_different_salt():
    signer1 = Signer("secret-key", salt="salt1", sep=".")
    signer2 = Signer("secret-key", salt="salt2", sep=".")
    value = b"test_value"
    signature = signer1.get_signature(value)
    assert signer2.verify_signature(value, signature) == False

def test_verify_signature_bytes_value():
    signer = Signer("secret-key", salt="salt", sep=".")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_different_separator():
    signer = Signer("secret-key", salt="salt", sep=":")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True


# LLM-generated content at query #11
#--------------------------

def test_verify_signature_with_valid_base64_sig_does_not_return_false():
    signer = Signer(secret_key="test-secret")
    sig = signer.get_signature(b"test-value")
    result = signer.verify_signature(b"test-value", sig)
    assert result is True


# LLM-generated content at query #12
#--------------------------

def test_verify_signature_predicate_false():
    signer = Signer(secret_key="test-secret")
    # Provide an invalid base64 string that does not raise an exception
    # but still fails verification, so the predicate at line 5 is False
    result = signer.verify_signature(b"test-value", b"invalid-base64!")
    assert result == False


####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + DeepSeek t=0.8)        #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"test_value"
    assert signer.verify_signature(value, b"invalid_signature") == False

def test_verify_signature_with_empty_value():
    signer = Signer("secret-key")
    signature = signer.get_signature(b"")
    assert signer.verify_signature(b"", signature) == True

def test_verify_signature_with_empty_signature():
    signer = Signer("secret-key")
    value = b"test_value"
    assert signer.verify_signature(value, b"") == False

def test_verify_signature_with_unicode_value():
    signer = Signer("secret-key")
    value = "üñíçödé"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_unicode_signature():
    signer = Signer("secret-key")
    value = b"test_value"
    signature = signer.get_signature(value)
    unicode_sig = signature.decode("ascii")
    assert signer.verify_signature(value, unicode_sig) == True

def test_verify_signature_with_key_rotation():
    signer = Signer(["old_key", "new_key"])
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_old_key():
    signer = Signer(["old_key", "new_key"])
    value = b"test_value"
    old_signer = Signer("old_key")
    old_signature = old_signer.get_signature(value)
    assert signer.verify_signature(value, old_signature) == True

def test_verify_signature_with_different_sep():
    signer = Signer("secret-key", sep=b"|")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_different_salt():
    signer = Signer("secret-key", salt=b"custom_salt")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_different_digest_method():
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_different_key_derivation():
    signer = Signer("secret-key", key_derivation="hmac")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_none_key_derivation():
    signer = Signer("secret-key", key_derivation="none")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_wrong_key():
    signer = Signer("secret-key")
    other_signer = Signer("wrong-key")
    value = b"test_value"
    wrong_signature = other_signer.get_signature(value)
    assert signer.verify_signature(value, wrong_signature) == False

def test_verify_signature_with_bytes_value():
    signer = Signer(b"secret-key")
    value = b"test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_string_value():
    signer = Signer("secret-key")
    value = "test_value"
    signature = signer.get_signature(value)
    assert signer.verify_signature(value, signature) == True

def test_verify_signature_with_base64_decode_error():
    signer = Signer("secret-key")
    value = b"test_value"
    assert signer.verify_signature(value, b"!!!invalid_base64!!!") == False


# LLM-generated content at query #2
#--------------------------

def test_signer_constructor_defaults():
    signer = Signer("secret-key")
    assert signer.secret_keys == [b"secret-key"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert signer.digest_method is not None
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_salt():
    signer = Signer("secret-key", salt=b"custom-salt")
    assert signer.salt == b"custom-salt"

def test_signer_constructor_with_sep():
    signer = Signer("secret-key", sep=b":")
    assert signer.sep == b":"

def test_signer_constructor_with_key_derivation():
    signer = Signer("secret-key", key_derivation="concat")
    assert signer.key_derivation == "concat"

def test_signer_constructor_with_digest_method():
    import hashlib
    signer = Signer("secret-key", digest_method=hashlib.sha256)
    assert signer.digest_method is hashlib.sha256

def test_signer_constructor_with_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret-key", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_with_secret_key_list():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_sep_in_base64_alphabet_raises():
    from itsdangerous.exc import BadSignature
    try:
        Signer("secret-key", sep=b".")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"

def test_signer_constructor_salt_none():
    signer = Signer("secret-key", salt=None)
    assert signer.salt == b"itsdangerous.Signer"


# LLM-generated content at query #3
#--------------------------

```python
def test_sep_in_base64_alphabet_raises_value_error():
    sep = b"a"
    from itsdangerous.signer import Signer
    secret_key = b"secret"

    try:
        signer = Signer(secret_key=secret_key, sep=sep)
        assert False, "Expected ValueError"
    except ValueError:
        pass
```


# LLM-generated content at query #4
#--------------------------

def test_verify_signature_with_invalid_base64_returns_false():
    signer = Signer(secret_key=b"test-secret")
    result = signer.verify_signature(b"test-value", b"!!!invalid-base64!!!")
    assert result == False


# LLM-generated content at query #5
#--------------------------

def test_verify_signature_with_valid_signature():
    signer = Signer(secret_key="secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_invalid_signature():
    signer = Signer(secret_key="secret-key")
    value = b"test value"
    invalid_sig = b"invalid_signature"
    assert signer.verify_signature(value, invalid_sig) == False

def test_verify_signature_with_empty_value():
    signer = Signer(secret_key="secret-key")
    value = b""
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_bytes_value():
    signer = Signer(secret_key="secret-key")
    value = b"test bytes"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_string_value():
    signer = Signer(secret_key="secret-key")
    value = "test string"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_multiple_secret_keys():
    signer = Signer(secret_key=["old-key", "new-key"])
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_old_secret_key():
    signer = Signer(secret_key=["old-key", "new-key"])
    value = b"test value"
    sig = signer.get_signature(value)
    signer.secret_keys = [b"old-key"]
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_wrong_secret_key():
    signer = Signer(secret_key="secret-key")
    different_signer = Signer(secret_key="different-key")
    value = b"test value"
    sig = different_signer.get_signature(value)
    assert signer.verify_signature(value, sig) == False

def test_verify_signature_with_non_base64_sig():
    signer = Signer(secret_key="secret-key")
    value = b"test value"
    non_base64_sig = b"not_base64!!!"
    assert signer.verify_signature(value, non_base64_sig) == False

def test_verify_signature_with_bytes_sig():
    signer = Signer(secret_key="secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True


# LLM-generated content at query #6
#--------------------------

def test_signer_constructor_default_parameters():
    signer = Signer("secret")
    assert signer.secret_keys == [b"secret"]
    assert signer.sep == b"."
    assert signer.salt == b"itsdangerous.Signer"
    assert signer.key_derivation == "django-concat"
    assert isinstance(signer.digest_method, type)
    assert isinstance(signer.algorithm, HMACAlgorithm)

def test_signer_constructor_with_bytes_secret_key():
    signer = Signer(b"secret")
    assert signer.secret_keys == [b"secret"]

def test_signer_constructor_with_list_of_secret_keys():
    signer = Signer(["key1", "key2"])
    assert signer.secret_keys == [b"key1", b"key2"]

def test_signer_constructor_with_custom_salt():
    signer = Signer("secret", salt="custom_salt")
    assert signer.salt == b"custom_salt"

def test_signer_constructor_with_custom_separator():
    signer = Signer("secret", sep=":")
    assert signer.sep == b":"

def test_signer_constructor_with_custom_key_derivation():
    signer = Signer("secret", key_derivation="hmac")
    assert signer.key_derivation == "hmac"

def test_signer_constructor_with_custom_digest_method():
    import hashlib
    signer = Signer("secret", digest_method=hashlib.sha256)
    assert signer.digest_method == hashlib.sha256

def test_signer_constructor_with_custom_algorithm():
    algorithm = HMACAlgorithm()
    signer = Signer("secret", algorithm=algorithm)
    assert signer.algorithm is algorithm

def test_signer_constructor_separator_in_base64_alphabet_raises_error():
    import pytest
    with pytest.raises(ValueError):
        Signer("secret", sep="a")

def test_signer_constructor_secret_key_property():
    signer = Signer(["key1", "key2"])
    assert signer.secret_key == b"key2"


# LLM-generated content at query #7
#--------------------------

def test_verify_signature_returns_false_when_base64_decode_raises_exception():
    signer = Signer(secret_key="secret-key")
    value = b"test_value"
    sig = "invalid_base64!!!"
    result = signer.verify_signature(value, sig)
    assert result is False


# LLM-generated content at query #8
#--------------------------

def test_verify_signature_exception_not_raised():
    signer = Signer("secret-key")
    valid_sig = signer.get_signature(b"test")
    result = signer.verify_signature(b"test", valid_sig)
    assert result == True


# LLM-generated content at query #9
#--------------------------

def test_verify_signature_with_valid_signature():
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_invalid_signature():
    signer = Signer("secret-key")
    value = b"test value"
    sig = b"invalid_signature"
    assert signer.verify_signature(value, sig) == False

def test_verify_signature_with_non_base64_signature():
    signer = Signer("secret-key")
    value = b"test value"
    sig = b"!!!not base64!!!"
    assert signer.verify_signature(value, sig) == False

def test_verify_signature_with_empty_signature():
    signer = Signer("secret-key")
    value = b"test value"
    sig = b""
    assert signer.verify_signature(value, sig) == False

def test_verify_signature_with_string_value():
    signer = Signer("secret-key")
    value = "test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_string_signature():
    signer = Signer("secret-key")
    value = b"test value"
    sig = signer.get_signature(value).decode("ascii")
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_multiple_secret_keys():
    signer = Signer(["old-key", "new-key"])
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_old_secret_key():
    signer = Signer(["old-key", "new-key"])
    value = b"test value"
    old_signer = Signer("old-key")
    sig = old_signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_wrong_secret_key():
    signer = Signer("secret-key")
    wrong_signer = Signer("wrong-key")
    value = b"test value"
    sig = wrong_signer.get_signature(value)
    assert signer.verify_signature(value, sig) == False

def test_verify_signature_with_different_salt():
    signer = Signer("secret-key", salt=b"custom-salt")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_different_separator():
    signer = Signer("secret-key", sep=b"|")
    value = b"test value"
    sig = signer.get_signature(value)
    assert signer.verify_signature(value, sig) == True

def test_verify_signature_with_none_value():
    signer = Signer("secret-key")
    sig = signer.get_signature(b"")
    assert signer.verify_signature(b"", sig) == True


# LLM-generated content at query #10
#--------------------------

def test_verify_signature_returns_false_when_base64_decode_raises_exception():
    signer = Signer(secret_key=b"test-secret")
    value = b"test-value"
    sig = "invalid base64!!!"
    result = signer.verify_signature(value, sig)
    assert result == False


# LLM-generated content at query #11
#--------------------------

def test_verify_signature_decoding_exception_returns_false():
    signer = Signer(secret_key="secret-key", salt="salt")
    result = signer.verify_signature(b"test", b"!!!invalid-base64!!!")
    assert result is False


