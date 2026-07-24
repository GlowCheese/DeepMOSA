####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Signer_unsign():
    # Setup common variables
    secret_key = b"secret"
    salt = b"salt"
    sep = b"."
    payload = b"hello-world"
    
    # Create a Signer instance
    signer = Signer(secret_key=secret_key, salt=salt, sep=sep)

    # 1. Test successful unsign
    signed_value = signer.sign(payload)
    assert signer.unsign(signed_value) == payload

    # 2. Test unsign with string input (should work due to want_bytes)
    signed_value_str = signer.sign("hello-world")
    assert signer.unsign(signed_value_str) == b"hello-world"

    # 3. Test BadSignature when separator is missing
    with pytest.raises(BadSignature) as excinfo:
        signer.unsign(b"no-separator-here")
    assert "No b'.'" in str(excinfo.value)

    # 4. Test BadSignature when signature does not match (tampered payload)
    tampered_payload = b"tampered-payload"
    signature = signer.get_signature(payload)
    # Construct a validly formatted but incorrect signed string: payload + sep + wrong_sig
    bad_signed_value = tampered_payload + sep + signature
    with pytest.raises(BadSignature) as excinfo:
        signer.unsign(bad_signed_value)
    assert "does not match" in str(excinfo.value)
    # Ensure the payload part of the exception is the original (tampered) payload
    assert excinfo.value.payload == tampered_payload

    # 5. Test BadSignature when signature is malformed (not base64)
    malformed_sig_value = payload + sep + b"!!!" 
    with pytest.raises(BadSignature):
        signer.unsign(malformed_sig_value)

    # 6. Test key rotation: verify signature works with an older key in the list
    old_key = b"old-secret"
    new_key = b"new-secret"
    rotational_signer = Signer(secret_key=[old_key, new_key], salt=salt, sep=sep)
    
    # Sign with the newest key (new_key)
    signed_with_new = rotational_signer.sign(payload)
    assert rotational_signer.unsign(signed_with_new) == payload

    # Manually create a signed value using the old key
    old_derived_key = rotational_signer.derive_key(old_key)
    old_sig = rotational_signer.algorithm.get_signature(old_derived_key, payload)
    from .encoding import base64_encode
    signed_with_old = payload + sep + base64_encode(old_sig)
    
    # Should still be able to unsign because old_key is in the list
    assert rotational_signer.unsign(signed_with_old) == payload

    # 7. Test with custom separator (that isn't in base64 alphabet)
    custom_sep = b"|"
    custom_signer = Signer(secret_key=secret_key, salt=salt, sep=custom_sep)
    signed_custom = custom_signer.sign(payload)
    assert custom_signer.unsign(signed_custom) == payload

    # 8. Test ValueError for invalid separator (e.g., 'A' is in base64 alphabet)
    with pytest.raises(ValueError) as excinfo:
        Signer(secret_key=secret_key, sep=b"A")
    assert "cannot be used" in str(excinfo.value)
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Signer_unsign():
    # Setup common parameters
    secret_key = b"secret"
    salt = b"salt"
    sep = b"."
    value = b"payload"
    
    # We use a real Signer instance but we can mock the algorithm or keys if needed.
    # However, for unsign, testing with a real HMACAlgorithm is most robust.
    signer = Signer(secret_key=secret_key, salt=salt, sep=sep)
    
    # 1. Test successful unsigning
    signed_value = signer.sign(value)
    assert signer.unsign(signed_value) == value

    # 2. Test unsigning with different types (str instead of bytes)
    signed_value_str = signer.sign("payload_string")
    assert signer.unsign(signed_value_str) == b"payload_string"

    # 3. Test BadSignature when separator is missing
    with pytest.raises(BadSignature) as excinfo:
        signer.unsign(b"nosetere")
    assert "No b'.'" in str(excinfo.value)

    # 4. Test BadSignature when signature is invalid (tampered payload)
    # We take a valid signed value and change the payload part
    valid_signed = signer.sign(b"original_payload")
    parts = valid_signed.rsplit(sep, 1)
    tampered_value = b"tampered_payload" + sep + parts[1]
    with pytest.raises(BadSignature) as excinfo:
        signer.unsign(tampered_value)
    assert "Signature" in str(excinfo.value)
    assert excinfo.value.payload == b"tampered_payload"

    # 5. Test BadSignature when signature is invalid (tampered signature)
    valid_signed = signer.sign(b"original_payload")
    parts = valid_signed.rsplit(sep, 1)
    # Modify the base64 signature part slightly
    sig_bytes = bytearray(parts[1])
    if len(sig_bytes) > 0:
        sig_bytes[0] = (sig_bytes[0] + 1) % 256
    tampered_sig_value = parts[0] + sep + bytes(sig_bytes)
    
    with pytest.raises(BadSignature):
        signer.unsign(tampered_sig_value)

    # 6. Test key rotation: unsigning with an old key in the list
    old_key = b"old_secret"
    new_key = b"new_secret"
    rotation_signer = Signer(secret_key=[old_key, new_key], salt=salt)
    
    # Sign with the newest key (new_key)
    signed_with_new = rotation_signer.sign(b"data")
    # Should work because rotation_signer knows both keys
    assert rotation_signer.unsign(signed_with_new) == b"data"
    
    # Manually create a signed value using the old key
    # We need to derive the key manually for this specific test case logic
    old_derived_key = rotation_signer.derive_key(old_key)
    sig_for_old = base64_encode(rotation_signer.algorithm.get_signature(old_derived_key, b"data"))
    signed_with_old = b"data" + sep + sig_for_old
    
    # Should be valid because the signer iterates through all keys in rotation
    assert rotation_signer.unsign(signed_with_old) == b"data"

    # 7. Test with invalid base64 signature string
    bad_b64_sig = b"data" + sep + b"!!!NotBase64!!!"
    with pytest.raises(BadSignature):
        signer.unsign(bad_b64_sig)
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = digest_method(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_conct
    assert signer_concat.derive_key(b"other") == digest_method(salt + b"other").digest()

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = digest_method(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django
    assert signer_django.derive_key(b"other") == digest_method(salt + b"signer" + b"other").digest()

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    expected_hmac = hmac.new(secret_key, digestmod=digest_method)
    expected_hmac.update(salt)
    assert signer_hmac.derive_key() == expected_hmac.digest()
    
    expected_hmac_other = hmac.new(b"other", digestmod=digest_method)
    expected_hmac_other.update(salt)
    assert signer_hmac.derive_key(b"other") == expected_hmac_other.digest()

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key
    assert signer_none.derive_key(b"other") == b"other"

    # Test rotation (using list of keys)
    keys = [b"old", b"new"]
    signer_rotation = Signer(
        secret_key=keys, 
        salt=salt, 
        key_derivation="none"
    )
    # derive_key() without args should use the newest (last) key
    assert signer_rotation.derive_key() == b"new"
    # Explicitly passing an old key from the list
    assert signer_rotation.derive_key(b"old") == b"old"

    # Test TypeError for unknown method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid_method")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Signer_verify_signature():
    """Tests the verify_signature method of the Signer class."""
    secret_key = b"secret"
    salt = b"salt"
    sep = b"."
    value = b"payload"
    
    # Setup a valid signer and signature manually for testing logic
    signer = Signer(secret_key=secret_key, salt=salt, sep=sep)
    signature = signer.get_signature(value)
    signed_value_with_sig = value + sep + signature

    # 1. Test Valid Signature
    # The signature is passed as base64 encoded bytes/str in the method
    assert signer.verify_signature(value, signature) is True
    assert signer.verify_signature(value, signature.decode()) is True
    assert signer.verify_signature(value, signed_value_with_sig.split(sep)[1]) is True

    # 2. Test Invalid Signature (Tampered Payload)
    tampered_value = b"tampered_payload"
    assert signer.verify_signature(tampered_value, signature) is False

    # 3. Test Invalid Signature (Incorrect Key/Signature)
    wrong_signature = base64_encode(b"wrong_sig")
    assert signer.verify_signature(value, wrong_signature) is False

    # 4. Test Key Rotation (Validating with an older key in the list)
    old_key = b"old_secret"
    new_key = b"new_secret"
    # Signer initialized with [old, new]. Newest (last) is used for signing.
    rotating_signer = Signer(secret_key=[old_key, new_key], salt=salt, sep=sep)
    
    # Generate signature using the newest key (new_key)
    valid_sig_new = rotating_signer.get_signature(value)
    # Verify it works with the current signer state
    assert rotating_signer.verify_signature(value, valid_sig_new) is True

    # Manually create a signature using the old key to see if rotation verification works
    # We must derive the key for the old key manually to simulate an old valid signature
    old_key_derived = rotating_signer.derive_key(old_key)
    # Use HMACAlgorithm directly as the Signer would have done internally
    algo = HMACAlgorithm()
    valid_sig_old = base64_encode(algo.get_signature(old_key_derived, value))
    
    # This should return True because verify_signature iterates through reversed(secret_keys)
    assert rotating_signer.verify_signature(value, valid_sig_old) is True

    # 5. Test Malformed Base64 Signature
    # Passing something that cannot be base64 decoded
    assert signer.verify_signature(value, b"!!!NotBase64!!!") is False

    # 6. Test with Custom Algorithm (NoneAlgorithm)
    none_algo = NoneAlgorithm()
    signer_none = Signer(secret_key=secret_key, algorithm=none_algo)
    # NoneAlgorithm returns empty bytes. Base64 of empty is empty string/bytes.
    empty_sig = base64_encode(b"")
    assert signer_none.verify_signature(value, empty_sig) is True
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret = b"secret"
    salt = b"salt"
    
    # Test 'concat' derivation
    signer_concat = Signer(secret_key=secret, salt=salt, key_derivation="concat")
    expected_concat = hashlib.sha1(salt + secret).digest()
    assert signer_can_derive_key(signer_concat) == expected_concat

    # Test 'django-concat' derivation
    signer_django = Signer(secret_key=secret, salt=salt, key_derivation="django-concat")
    expected_django = hashlib.sha1(salt + b"signer" + secret).digest()
    assert signer_can_derive_key(signer_django) == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(secret_key=secret, salt=salt, key_derivation="hmac")
    expected_hmac = hmac.new(secret, msg=salt, digestmod=hashlib.sha1).digest()
    assert signer_can_derive_key(signer_hmac) == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(secret_key=secret, salt=salt, key_derivation="none")
    assert signer_can_derive_key(signer_none) == secret

    # Test with explicit secret_key passed to derive_key
    alternative_secret = b"alt_secret"
    signer_concat_alt = Signer(secret_key=secret, salt=salt, key_derivation="concat")
    expected_alt = hashlib.sha1(salt + alternative_secret).digest()
    assert signer_concat_alt.derive_key(secret_key=alternative_secret) == expected_alt

    # Test with rotation (using the newest key from the list)
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rotation.derive_key() == b"new"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

def signer_can_derive_key(signer):
    """Helper to call the method for testing."""
    return signer.derive_key()
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret = b"secret"
    salt = b"salt"
    
    # Test 'concat' derivation
    signer_concat = Signer(secret_key=secret, salt=salt, key_derivation="concat")
    expected_concat = hashlib.sha1(salt + secret).digest()
    assert signer_can_derive_key(signer_concat, secret) == expected_concat
    assert signer_can_derive_key(signer_concat) == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(secret_key=secret, salt=salt, key_derivation="django-concat")
    expected_django = hashlib.sha1(salt + b"signer" + secret).digest()
    assert signer_can_derive_key(signer_django, secret) == expected_django
    assert signer_can_derive_key(signer_django) == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(secret_key=secret, salt=salt, key_derivation="hmac")
    expected_hmac = hmac.new(secret, digestmod=hashlib.sha1).update(salt) or hmac.new(secret, salt, hashlib.sha1).digest()
    # Note: The implementation uses mac.update(salt), so:
    h = hmac.new(secret, msg=salt, digestmod=hashlib.sha1)
    expected_hmac = h.digest()
    assert signer_can_derive_key(signer_hmac, secret) == expected_hmac
    assert signer_can_derive_key(signer_hmac) == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(secret_key=secret, salt=salt, key_derivation="none")
    assert signer_can_derive_key(signer_none, secret) == secret
    assert signer_can_derive_key(signer_none) == secret

    # Test with explicit different secret_key passed to derive_key
    other_secret = b"other"
    assert signer_can_derive_key(signer_concat, other_secret) == hashlib.sha1(salt + other_secret).digest()

    # Test rotation (multiple keys)
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rotation.secret_key == b"new"
    assert signer_rotation.derive_key(b"old") == b"old"
    assert signer_rotation.derive_key() == b"new"

    # Test error for unknown derivation
    signer_invalid = Signer(secret_key=secret, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

def signer_can_derive_key(signer, secret):
    return signer.derive_key(secret)
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = digest_method(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_conat
    assert signer_concat.derive_key(b"other") == digest_method(salt + b"other").digest()

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = digest_method(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django
    assert signer_django.derive_key(b"other") == digest_method(salt + b"signer" + b"other").digest()

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    mac = hmac.new(secret_key, digestmod=digest_method)
    mac.update(salt)
    expected_hmac = mac.digest()
    assert signer_hmac.derive_key() == expected_hmac
    
    mac_other = hmac.new(b"other", digestmod=digest_method)
    mac_other.update(salt)
    assert signer_hmac.derive_key(b"other") == mac_other.digest()

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key
    assert signer_none.derive_key(b"other") == b"other"

    # Test Invalid derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

    # Test rotation (passing list of keys)
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rotation.derive_key() == b"new"
    assert signer_rotation.derive_key(b"old") == b"old"
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key,
        salt=salt,
        key_derivation="concat",
        digest_method=digest_method
    )
    expected_concat = hashlib.sha256(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_canv_concat := expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key,
        salt=salt,
        key_derivation="django-concat",
        digest_method=digest_method
    )
    expected_django = hashlib.sha256(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key,
        salt=salt,
        key_derivation="hmac",
        digest_method=digest_method
    )
    expected_hmac = hmac.new(secret_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key,
        salt=salt,
        key_derivation="none",
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key

    # Test passing a specific secret_key to derive_key
    alt_key = b"alternative"
    signer_alt = Signer(
        secret_key=secret_key,
        salt=salt,
        key_derivation="concat",
        digest_method=digest_method
    )
    expected_alt = hashlib.sha256(salt + alt_key).digest()
    assert signer_alt.derive_key(secret_key=alt_key) == expected_alt

    # Test key rotation (using list of keys)
    keys = [b"old", b"new"]
    signer_rotation = Signer(
        secret_key=keys,
        salt=salt,
        key_derivation="none"
    )
    # derive_key() should use the newest (last) key by default
    assert signer_rotation.derive_key() == b"new"
    # Manually providing an older key from the list
    assert signer_rotation.derive_key(secret_key=b"old") == b"old"

    # Test error for unknown derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid_method")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Signer_verify_signature():
    # Setup common variables
    secret_key = b"secret"
    salt = b"salt"
    value = b"data"
    sep = b"."
    signer = Signer(secret_key=secret_key, salt=salt, sep=sep)

    # 1. Test successful verification with correct signature
    signature = signer.get_signature(value)
    signed_payload = value + sep + signature
    # We extract the signature part from the signed payload for testing verify_signature directly
    sig_part = signed_payload.split(sep)[1]
    assert signer.verify_signature(value, sig_part) is True

    # 2. Test failure with incorrect signature
    wrong_signature = b"bm90X3RoZV9zaWduYXR1cmU=" # base64 for "not_the_signature"
    assert signer.verify_signature(value, wrong_signature) is False

    # 3. Test failure with corrupted value
    corrupted_value = b"different_data"
    assert signer.verify_signature(corrupted_value, sig_part) is False

    # 4. Test failure with malformed base64 signature
    malformed_sig = b"!!!not_base64!!!"
    assert signer.verify_signature(value, malformed_sig) is False

    # 5. Test key rotation support (verifying with an older key in the list)
    old_key = b"old_secret"
    signer_with_rotation = Signer(secret_key=[old_key, secret_key], salt=salt)
    
    # Generate signature using the OLD key manually
    # We must replicate the derivation logic for the old key
    old_derived_key = signer_with_rotation.derive_key(old_key)
    algo = HMACAlgorithm()
    old_sig_raw = algo.get_signature(old_derived_key, value)
    old_sig_b64 = base64_encode(old_sig_raw)
    
    # The newest key (secret_key) is used for signing in the class, 
    # but verify_signature should check all keys in reversed order.
    assert signer_with_rotation.verify_signature(value, old_sig_b64) is True

    # 6. Test with a completely different algorithm (NoneAlgorithm)
    none_algo = NoneAlgorithm()
    signer_none = Signer(secret_key=secret_key, algorithm=none_algo)
    # NoneAlgorithm returns b"", so signature is empty base64
    empty_sig = b"" 
    assert signer_none.verify_signature(value, empty_sig) is True
    assert signer_none.verify_signature(value, b"something") is False

    # 7. Test with custom key derivation 'hmac'
    signer_hmac = Signer(secret_key=secret_key, salt=salt, key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac.split(sep)[1] if sep in sig_hmac else sig_hmac) is True

    # 8. Test with custom separator
    custom_sep = b"|"
    signer_sep = Signer(secret_key=secret_key, sep=custom_sep)
    sig_sep = signer_sep.get_signature(value)
    # Verification should work via the split logic in the test if we provide the part after |
    parts = (value + custom_sep + sig_sep).split(custom_sep)
    assert signer_sep.verify_signature(value, parts[1]) is True
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'none' derivation
    signer_none = Signer(secret_key=secret_key, salt=salt, key_derivation="none", digest_method=digest_mode)
    assert signer_none.derive_key() == secret_key
    assert signer_none.derive_key(b"other") == b"other"

    # Test 'concat' derivation: digest(salt + secret)
    signer_concat = Signer(secret_key=secret_key, salt=salt, key_derivation="concat", digest_method=digest_method)
    expected_concat = hashlib.sha256(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat
    assert signer_concat.derive_key(b"other") == hashlib.sha256(salt + b"other").digest()

    # Test 'django-concat' derivation: digest(salt + b"signer" + secret)
    signer_django = Signer(secret_key=secret_key, salt=salt, key_derivation="django-concat", digest_method=digest_method)
    expected_django = hashlib.sha256(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django
    assert signer_django.derive_key(b"other") == hashlib.sha256(salt + b"signer" + b"other").digest()

    # Test 'hmac' derivation: hmac(secret, salt)
    signer_hmac = Signer(secret_key=secret_key, salt=salt, key_derivation="hmac", digest_method=digest_method)
    expected_hmac = hmac.new(secret_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key() == expected_hmac
    
    other_key = b"other_key"
    expected_hmac_other = hmac.new(other_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key(other_key) == expected_hmac_other

    # Test key rotation (using list of keys)
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    # Default should use the newest (last) key
    assert signer_rotation.derive_key() == b"new"
    # Explicitly passing an older key from the list
    assert signer_rotation.derive_key(b"old") == b"old"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid_method")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = hashlib.sha256(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = hashlib.sha256(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    mac = hmac.new(secret_key, digestmod=digest_method)
    mac.update(salt)
    expected_hmac = mac.digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key

    # Test passing specific secret_key to derive_key
    alt_secret = b"alt_secret"
    assert signer_concat.derive_key(secret_key=alt_secret) == hashlib.sha256(salt + alt_secret).digest()

    # Test error on unknown derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

    # Test with list of keys (rotation) - should use the provided key if passed, or latest if not
    keys = [b"old", b"new"]
    signer_rot = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rot.derive_key() == b"new"
    assert signer_rot.derive_key(secret_key=b"old") == b"old"
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest

def test_Signer_verify_signature():
    secret = b"secret-key"
    salt = b"test-salt"
    signer = Signer(secret_key=secret, salt=salt)
    value = b"hello-world"
    
    # Generate a valid signature using the signer's own sign method
    signed_value = signer.sign(value)
    signature = signed_value.rsplit(signer.sep, 1)[1]

    # Case 1: Valid signature
    assert signer.verify_signature(value, signature) is True

    # Case 2: Invalid signature (tampered value)
    tampered_value = b"hello-worle"
    assert signer.verify_signature(tampered_value, signature) is False

    # Case 3: Invalid signature (tampered signature bytes)
    # We decode the base64, change a byte, and re-encode to simulate valid B64 but wrong content
    from .encoding import base64_decode, base64_encode
    decoded_sig = bytearray(base64_decode(signature))
    decoded_sig[0] ^= 0xFF  # Flip bits in the first byte
    tampered_signature = base64_encode(bytes(decoded_sig))
    assert signer.verify_signature(value, tampered_signature) is False

    # Case 4: Invalid signature (malformed base64)
    assert signer.verify_signature(value, b"!!!not-base64!!!") is False

    # Case 5: Key rotation support
    old_secret = b"old-key"
    new_secret = b"new-key"
    rotation_signer = Signer(secret_key=[old_secret, new_secret], salt=salt)
    
    # Signature created with the old key should still be verifiable
    old_signed_value = rotation_signer.sign(value) # This uses newest (new_secret)
    # To test old key, we manually use the algorithm for the old key
    key_old = rotation_signer.derive_key(old_secret)
    sig_old_raw = rotation_signer.algorithm.get_signature(key_old, value)
    sig_old_b64 = base64_encode(sig_old_raw)
    
    assert rotation_signer.verify_signature(value, sig_old_b64) is True

    # Case 6: Signature from a completely different signer/key
    other_signer = Signer(secret_key=b"different-key", salt=salt)
    other_signature = other_signer.sign(value).rsplit(other_signer.sep, 1)[1]
    assert signer.verify_signature(value, other_signature) is False
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = digest_method(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_contra
    
    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = digest_method(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    expected_hmac = hmac.new(secret_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key

    # Test providing a specific secret_key to derive_key method (using django-concat default)
    alternate_key = b"new_key"
    expected_alt = digest_method(salt + b"signer" + alternate_key).digest()
    assert signer_django.derive_key(secret_key=alternate_key) == expected_alt

    # Test rotation: derive_key should use the specific key from the list if passed
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rotation.derive_key(secret_key=b"old") == b"old"
    assert signer_rotation.derive_key(secret_key=b"new") == b"new"

    # Test invalid derivation method raises TypeError
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid_method")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Signer_verify_signature():
    secret_key = b"secret"
    salt = b"salt"
    sep = b"."
    value = b"payload"
    
    # 1. Test successful verification with correct signature
    signer = Signer(secret_key=secret_key, salt=salt, sep=sep)
    signature = signer.get_signature(value)
    signed_value = value + sep + signature
    # Extracting the sig part from the signed string for direct testing of verify_signature
    sig_part = signed_value.split(sep)[1]
    
    assert signer.verify_signature(value, sig_part) is True

    # 2. Test failure with incorrect signature
    wrong_signature = b"bm90X3RoZV9zaWduYXR1cmU=" # base64 for "not_the_signature"
    assert signer.verify_signature(value, wrong_signature) is False

    # 3. Test failure with malformed base64 signature
    assert signer.verify_signature(value, b"!!!") is False

    # 4. Test key rotation (verifying signature created with an older key)
    old_key = b"old_secret"
    new_key = b"new_secret"
    signer_rotation = Signer(secret_key=[old_key, new_key], salt=salt, sep=sep)
    
    # Signature created with the old key (the first in list)
    # Note: get_signature uses the newest key (last in list), 
    # so we must manually simulate the older key derivation for this test.
    old_derived_key = signer_rotation.derive_key(old_key)
    algorithm = HMACAlgorithm()
    old_sig_raw = algorithm.get_signature(old_derived_key, value)
    old_sig_b64 = base64_encode(old_sig_raw)
    
    # Should return True because verify_signature iterates through all keys
    assert signer_rotation.verify_signature(value, old_sig_b64) is True

    # 5. Test failure with completely different value
    different_value = b"different_payload"
    assert signer.verify_signature(different_value, sig_part) is False

    # 6. Test with custom algorithm (Mocking the algorithm behavior)
    mock_algo = MagicMock(spec=SigningAlgorithm)
    # Mock verify_signature to return True only for a specific input
    mock_algo.verify_signature.side_effect = lambda k, v, s: v == value and s == b"valid_sig"
    
    signer_mock = Signer(secret_key=secret_key, algorithm=mock_algo)
    # We provide a signature that when base64 decoded equals b"valid_sig"
    valid_b64_sig = base64_encode(b"valid_sig")
    
    assert signer_mock.verify_signature(value, valid_b64_sig) is True
    assert signer_mock.verify_signature(b"wrong_value", valid_b64_sig) is False
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256
    
    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = hashlib.sha256(salt + secret).digest()
    assert signer_concat.derive_key() == expected_concat
    assert signer_concat.derive_key(b"other") == hashlib.sha256(salt + b"other").digest()

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = hashlib.sha256(salt + b"signer" + secret).digest()
    assert signer_django.derive_key() == expected_django
    assert signer_django.derive_key(b"other") == hashlib.sha256(salt + b"signer" + b"other").digest()

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    mac = hmac.new(secret, digestmod=digest_method)
    mac.update(salt)
    expected_hmac = mac.digest()
    assert signer_hmac.derive_key() == expected_hmac
    
    mac_other = hmac.new(b"other", digestmod=digest_method)
    mac_other.update(salt)
    assert signer_hmac.derive_key(b"other") == mac_other.digest()

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret, 
        salt=salt, 
        key_derivation="none"
    )
    assert signer_none.derive_key() == secret
    assert signer_none.derive_key(b"other") == b"other"

    # Test key rotation (using list of keys)
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rotation.derive_key() == b"new"
    # verify_signature uses reversed loop internally, but derive_key(key) 
    # should allow manual derivation of the specific old key
    assert signer_rotation.derive_key(b"old") == b"old"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret, key_derivation="invalid_method")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest

def test_Signer_verify_signature():
    secret = b"secret-key"
    salt = b"test-salt"
    signer = Signer(secret_key=secret, salt=salt)
    value = b"hello-world"
    
    # Generate a valid signature using the signer's own logic
    signature = signer.get_signature(value)
    
    # 1. Test successful verification with correct signature
    assert signer.verify_signature(value, signature) is True
    
    # 2. Test failure with incorrect value
    wrong_value = b"wrong-value"
    assert signer.verify_signature(wrong_value, signature) is False
    
    # 3. Test failure with tampered signature (corrupted base64/bytes)
    tampered_sig = signature[:-1] + (b"A" if signature[-1:] != b"A" else b"B")
    assert signer.verify_signature(value, tampered_sig) is False
    
    # 4. Test failure with invalid base64 encoding
    invalid_b64 = b"!!!not-base64!!!"
    assert signer.verify_signature(value, invalid_b64) is False

    # 5. Test success with key rotation (verifying signature made by an old key)
    old_secret = b"old-key"
    old_signer = Signer(secret_key=[old_secret, secret], salt=salt)
    old_signature = old_signer.get_signature(value)
    # The new signer should be able to verify the signature using the 'old' key in its list
    assert signer.verify_signature(value, old_signature) is True

    # 6. Test failure with a completely different signer/key
    different_signer = Signer(secret_key=b"different-key", salt=salt)
    assert different_signer.verify_signature(value, signature) is False
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret = b"secret"
    salt = b"salt"
    digest = hashlib.sha1

    # Test 'concat' derivation
    signer_concat = Signer(secret_key=secret, salt=salt, key_derivation="concat", digest_method=digest)
    expected_concat = digest(salt + secret).digest()
    assert signer_can_derive_key(signer_concat, None) == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(secret_key=secret, salt=salt, key_derivation="django-concat", digest_method=digest)
    expected_django = digest(salt + b"signer" + secret).digest()
    assert signer_can_derive_key(signer_django, None) == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(secret_key=secret, salt=salt, key_derivation="hmac", digest_method=digest)
    mac = hmac.new(secret, digestmod=digest)
    mac.update(salt)
    expected_hmac = mac.digest()
    assert signer_can_derive_key(signer_hmac, None) == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(secret_key=secret, salt=salt, key_derivation="none")
    assert signer_can_derive_key(signer_none, None) == secret

    # Test passing an explicit secret_key to derive_key
    alt_secret = b"alt_secret"
    assert signer_concat.derive_key(alt_secret) == digest(salt + alt_secret).digest()

    # Test rotation (list of keys) - should use the provided key if passed, or last in list if None
    keys = [b"old", b"new"]
    signer_rot = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rot.derive_key(None) == b"new"
    assert signer_rot.derive_key(b"old") == b"old"

    # Test error for unknown derivation
    signer_invalid = Signer(secret_key=secret, key_derivation="unknown")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

def signer_can_derive_key(signer, explicit_key):
    return signer.derive_key(explicit_key)
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = hashlib.sha256(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = hashlib.sha256(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    expected_hmac = hmac.new(secret_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key

    # Test providing a specific secret_key to derive_key
    alt_key = b"alternative"
    assert signer_concat.derive_key(secret_key=alt_key) == hashlib.sha256(salt + alt_key).digest()
    assert signer_hmac.derive_key(secret_key=alt_key) == hmac.new(alt_key, msg=salt, digestmod=digest_method).digest()

    # Test key rotation (list of keys) - should use the provided key if passed, or latest if not
    keys = [b"old", b"new"]
    signer_rot = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rot.derive_key() == b"new"
    assert signer_rot.derive_key(secret_key=b"old") == b"old"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test "none" derivation
    signer_none = Signer(secret_key=secret_key, salt=salt, key_derivation="none", digest_method=digest_method)
    assert signer_none.derive_key() == secret_key
    assert signer_none.derive_key(b"other") == b"other"

    # Test "concat" derivation: digest(salt + secret)
    signer_concat = Signer(secret_key=secret_key, salt=salt, key_derivation="concat", digest_method=digest_method)
    expected_concat = hashlib.sha256(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat
    assert signer_concat.derive_key(b"other") == hashlib.sha256(salt + b"other").digest()

    # Test "django-concat" derivation: digest(salt + b"signer" + secret)
    signer_django = Signer(secret_key=secret_key, salt=salt, key_derivation="django-concat", digest_method=digest_method)
    expected_django = hashlib.sha256(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django
    assert signer_django.derive_key(b"other") == hashlib.sha256(salt + b"signer" + b"other").digest()

    # Test "hmac" derivation: hmac(secret, salt)
    signer_hmac = Signer(secret_key=secret_key, salt=salt, key_derivation="hmac", digest_method=digest_method)
    expected_hmac = hmac.new(secret_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key() == expected_hmac
    
    # Test hmac with specific key override
    other_key = b"another_secret"
    expected_hmac_other = hmac.new(other_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key(other_key) == expected_hmac_other

    # Test key rotation (using list of keys)
    keys = [b"old_key", b"new_key"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    # Should default to newest (last) key
    assert signer_rotation.derive_key() == b"new_key"
    # Should allow deriving from the oldest key explicitly
    assert signer_rotation.derive_key(b"old_key") == b"old_key"

    # Test error handling for unknown derivation
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid_method")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test "concat" derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = hashlib.sha256(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test "django-concat" derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = hashlib.sha256(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test "hmac" derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    expected_hmac = hmac.new(secret_key, digestmod=digest_method)
    expected_hmac.update(salt)
    assert signer_hmac.derive_key() == expected_hmac.digest()

    # Test "none" derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key

    # Test passing specific secret_key to derive_key
    alt_key = b"other-key"
    assert signer_concat.derive_key(secret_key=alt_key) == hashlib.sha256(salt + alt_key).digest()
    assert signer_none.derive_key(secret_key=alt_key) == alt_key

    # Test error on unknown derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

    # Test with list of keys (rotation) - should use the provided key if passed, or last key otherwise
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rotation.derive_key() == b"new"
    assert signer_rotation.derive_key(secret_key=b"old") == b"old"
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Signer_verify_signature():
    # Setup common variables
    secret_key = b"secret"
    salt = b"salt"
    value = b"payload"
    sep = b"."
    
    # 1. Test successful verification with correct signature
    signer = Signer(secret_key=secret_key, salt=salt, sep=sep)
    signature = signer.get_signature(value)
    signed_content = value + sep + signature
    # Extract just the sig part for verify_signature call
    sig_part = signed_content.split(sep)[1]
    
    assert signer.verify_signature(value, sig_part) is True

    # 2. Test failure with incorrect signature
    wrong_signature = b"bm90X3RoZV9zaWduYXR1cmU=" # base64 for "not_the_signature"
    assert signer.verify_signature(value, wrong_signature) is False

    # 3. Test failure with corrupted value (payload changed)
    tampered_value = b"tampered_payload"
    assert signer.verify_signature(tampered_value, sig_part) is False

    # 4. Test key rotation: verify signature created by an older key in the list
    old_key = b"old_secret"
    new_key = b"new_secret"
    signer_rotation = Signer(secret_key=[old_key, new_key], salt=salt, sep=sep)
    
    # Signature generated using the 'old' key (the first in list)
    # Note: sign() uses the newest key (last in list), so we must manually 
    # simulate the signature generation for the old key.
    old_derived_key = signer_rotation.derive_key(old_key)
    old_algorithm = HMACAlgorithm()
    old_sig_raw = old_algorithm.get_signature(old_derived_key, value)
    old_sig_b64 = base64_encode(old_sig_raw)
    
    # Should return True because verify_signature iterates through all keys
    assert signer_rotation.verify_signature(value, old_sig_b64) is True

    # 5. Test failure with completely invalid base64 input
    assert signer_rotation.verify_signature(value, b"!!!notbase64!!!") is False

    # 6. Test using a custom algorithm (NoneAlgorithm)
    signer_none = Signer(secret_key=secret_key, algorithm=NoneAlgorithm())
    sig_none = signer_none.get_signature(value)
    assert signer_none.verify_signature(value, sig_none) is True

    # 7. Test with different key derivation (hmac mode)
    signer_hmac = Signer(secret_key=secret_key, salt=salt, key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac) is True
```


####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = digest_method(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_conat
    assert signer_concat.derive_key(b"other") == digest_method(salt + b"other").digest()

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = digest_method(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django
    assert signer_django.derive_key(b"other") == digest_method(salt + b"signer" + b"other").digest()

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    mac = hmac.new(secret_key, digestmod=digest_method)
    mac.update(salt)
    expected_hmac = mac.digest()
    assert signer_hmac.derive_key() == expected_hmac
    
    mac_other = hmac.new(b"other", digestmod=digest_method)
    mac_other.update(salt)
    assert signer_hmac.derive_key(b"other") == mac_other.digest()

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key
    assert signer_none.derive_key(b"other") == b"other"

    # Test rotation (passing list of keys)
    keys = [b"old", b"new"]
    signer_rotation = Signer(
        secret_key=keys, 
        salt=salt, 
        key_derivation="none"
    )
    assert signer_rotation.derive_key() == b"new"  # Should use latest key
    assert signer_rotation.derive_key(b"old") == b"old"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #2
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = digest_method(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_conct
    assert signer_concat.derive_key(b"other") == digest_method(salt + b"other").digest()

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = digest_method(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django
    assert signer_django.derive_key(b"other") == digest_method(salt + b"signer" + b"other").digest()

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    expected_hmac = hmac.new(secret_key, digestmod=digest_method).digest()
    # Note: the implementation updates with salt: mac.update(salt)
    h = hmac.new(secret_key, digestmod=digest_method)
    h.update(salt)
    expected_hmac_actual = h.digest()
    assert signer_hmac.derive_key() == expected_hmac_actual
    
    h2 = hmac.new(b"other", digestmod=digest_method)
    h2.update(salt)
    assert signer_hmac.derive_key(b"other") == h2.digest()

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key
    assert signer_none.derive_key(b"other") == b"other"

    # Test rotation (using list of keys)
    keys = [b"old", b"new"]
    signer_rotation = Signer(
        secret_key=keys, 
        salt=salt, 
        key_derivation="none"
    )
    assert signer_rotation.derive_key() == b"new"
    assert signer_rotation.derive_key(b"old") == b"old"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #3
#--------------------------

```python
import pytest

def test_Signer_verify_signature():
    secret = b"super-secret-key"
    salt = b"test-salt"
    signer = Signer(secret_key=secret, salt=salt)
    payload = b"hello-world"
    
    # Generate a valid signature
    signature = signer.get_signature(payload)
    signed_value = payload + b"." + signature

    # Case 1: Valid signature
    assert signer.verify_signature(payload, signature) is True
    assert signer.verify_signature(payload, signed_value[len(payload)+1:]) is True
    
    # Case 2: Invalid signature (tampered payload)
    tampered_payload = b"tampered-world"
    assert signer.verify_signature(tampered_payload, signature) is False

    # Case 3: Invalid signature (tampered signature content)
    # We decode the base64, flip a bit, and re-encode to ensure it's valid B64 but wrong sig
    from .encoding import base64_decode, base64_encode
    raw_sig = base64_decode(signature)
    tampered_sig_bytes = bytearray(raw_sig)
    tampered_sig_bytes[0] ^= 0xFF 
    tampered_signature = base64_encode(bytes(tampered_sig_bytes))
    assert signer.verify_signature(payload, tampered_signature) is False

    # Case 4: Malformed Base64 signature
    assert signer.verify_signature(payload, b"not-base64-!!!") is False

    # Case 5: Key rotation (verifying with an older key in the list)
    old_secret = b"old-key"
    signer_rotation = Signer(secret_key=[old_secret, secret])
    # The signature was created with 'secret' (the newest/last key)
    # verify_signature should find it by iterating through keys
    assert signer_rotation.verify_signature(payload, signature) is True

    # Case 6: Signature from a completely different signer/key
    other_signer = Signer(secret_key=b"different-key", salt=salt)
    other_sig = other_signer.get_signature(payload)
    assert signer.verify_signature(payload, other_sig) is False

    # Case 7: Signature with different salt
    signer_diff_salt = Signer(secret_key=secret, salt=b"different-salt")
    assert signer_diff_salt.verify_signature(payload, signature) is False
```


# LLM-generated content at query #4
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest = hashlib.sha1

    # Test 'concat' derivation
    signer_concat = Signer(secret_key=secret_key, salt=salt, key_derivation="concat", digest_method=digest)
    expected_concat = digest(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(secret_key=secret_key, salt=salt, key_derivation="django-concat", digest_method=digest)
    expected_django = digest(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(secret_key=secret_key, salt=salt, key_derivation="hmac", digest_method=digest)
    mac = hmac.new(secret_key, msg=salt, digestmod=digest)
    expected_hmac = mac.digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(secret_key=secret_key, salt=salt, key_derivation="none")
    assert signer_none.derive_key() == secret_key

    # Test passing a specific secret_key to derive_key
    alt_key = b"alternative"
    signer_concat_alt = Signer(secret_key=secret_key, salt=salt, key_derivation="concat", digest_method=digest)
    expected_alt = digest(salt + alt_key).digest()
    assert signer_concat_alt.derive_key(secret_key=alt_key) == expected_alt

    # Test error on unknown derivation
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

    # Test rotation (using the last key in list)
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rotation.derive_key() == b"new"
    assert signer_rotation.derive_key(secret_key=b"old") == b"old"
```


# LLM-generated content at query #5
#--------------------------

```python
import pytest
import hmac
import hashlib

def test_Signer_verify_signature():
    secret_key = b"secret"
    salt = b"salt"
    sep = b"."
    value = b"payload"
    
    # Initialize Signer with standard settings
    signer = Signer(secret_key=secret_key, salt=salt, sep=sep)

    # 1. Test verification of a valid signature
    signature = signer.get_signature(value)
    signed_value = value + sep + signature
    # Extracting the sig part from signed_value to test verify_signature directly
    sig_part = signed_value.split(sep)[1]
    assert signer.verify_signature(value, sig_part) is True

    # 2. Test verification with an invalid signature (tampered payload)
    tampered_value = b"tampered_payload"
    assert signer.verify_signature(tampered_value, sig_part) is False

    # 3. Test verification with a completely different signature
    wrong_signature = signer.get_signature(b"different_value")
    assert signer.verify_signature(value, wrong_signature) is False

    # 4. Test verification with malformed base64 signature
    malformed_sig = b"!!!not-base64!!!"
    assert signer.verify_signature(value, malformed_sig) is False

    # 5. Test key rotation (verifying using an old key in the list)
    old_key = b"old_secret"
    new_key = b"new_secret"
    rotation_signer = Signer(secret_key=[old_key, new_key], salt=salt, sep=sep)
    
    # Create signature using the OLD key manually
    # We simulate what derive_key(old_key) would produce for 'django-concat'
    derived_old_key = rotation_signer.derive_key(old_key)
    algo = HMACAlgorithm(hashlib.sha1)
    old_sig_raw = algo.get_signature(derived_old_key, value)
    old_sig_b64 = base64_encode(old_sig_raw)

    # Verify that the signer recognizes the signature created by the old key
    assert rotation_signer.verify_signature(value, old_sig_b64) is True

    # 6. Test verification with a different derivation method (hmac)
    hmac_signer = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac"
    )
    hmac_sig = hmac_signer.get_signature(value)
    assert hmac_signer.verify_signature(value, hmac_sig) is True

    # 7. Test verification with a different algorithm (NoneAlgorithm)
    none_algo = NoneAlgorithm()
    none_signer = Signer(secret_key=secret_key, algorithm=none_algo)
    # NoneAlgorithm returns b"", so we provide an empty base64 string
    assert none_signer.verify_signature(value, b"") is True
```


# LLM-generated content at query #6
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest = hashlib.sha1

    # Test 'django-concat' (default)
    signer_django = Signer(secret_key=secret_key, salt=salt, key_derivation="django-concat", digest_method=digest)
    expected_django = digest(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'concat'
    signer_concat = Signer(secret_key=secret_key, salt=salt, key_derivation="concat", digest_method=digest)
    expected_concat = digest(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test 'hmac'
    signer_hmac = Signer(secret_key=secret_key, salt=salt, key_derivation="hmac", digest_method=digest)
    mac = hmac.new(secret_key, digestmod=digest)
    mac.update(salt)
    expected_hmac = mac.digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none'
    signer_none = Signer(secret_key=secret_key, salt=salt, key_derivation="none")
    assert signer_none.derive_key() == secret_key

    # Test passing specific secret_key to derive_key
    alt_key = b"alternative"
    signer_django_alt = Signer(secret_key=secret_key, salt=salt, key_derivation="django-concat", digest_method=digest)
    expected_alt = digest(salt + b"signer" + alt_key).digest()
    assert signer_django_alt.derive_key(secret_key=alt_key) == expected_alt

    # Test key rotation (using list of keys)
    keys = [b"old_key", b"new_key"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    # derive_key() uses the latest key by default
    assert signer_rotation.derive_key() == b"new_key"
    # verify we can explicitly derive using the older key from the list
    assert signer_rotation.derive_key(secret_key=b"old_key") == b"old_key"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid_method")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #7
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret = b"secret"
    salt = b"salt"
    digest = hashlib.sha1

    # Test 'django-concat' (default)
    signer_django = Signer(secret_key=secret, salt=salt, key_derivation="django-concat", digest_method=digest)
    expected_django = digest(salt + b"signer" + secret).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'concat'
    signer_concat = Signer(secret_key=secret, salt=salt, key_derivation="concat", digest_method=digest)
    expected_concat = digest(salt + secret).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test 'hmac'
    signer_hmac = Signer(secret_key=secret, salt=salt, key_derivation="hmac", digest_method=digest)
    expected_hmac = hmac.new(secret, msg=salt, digestmod=digest).digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none'
    signer_none = Signer(secret_key=secret, salt=flag, key_derivation="none")
    assert signer_none.derive_key() == secret

    # Test with specific secret_key passed to derive_key
    alt_secret = b"alternative"
    assert signer_django.derive_key(secret_key=alt_secret) == digest(salt + b"signer" + alt_secret).digest()

    # Test key rotation (using list of keys)
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    # Should use the newest (last) key by default
    assert signer_rotation.derive_key() == b"new"
    # Should allow deriving using the old key explicitly
    assert signer_rotation.derive_key(secret_key=b"old") == b"old"

    # Test TypeError for unknown derivation
    signer_invalid = Signer(secret_key=secret, key_derivation="invalid_method")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #8
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret = b"secret-key"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test "none" derivation
    signer_none = Signer(secret_key=secret, salt=salt, key_derivation="none", digest_mode=digest_method)
    # Manually adjusting because the provided code uses 'digest_method' in __init__
    signer_none.digest_method = digest_method 
    assert signer_none.derive_key() == secret
    assert signer_none.derive_key(b"other") == b"other"

    # Test "concat" derivation: hash(salt + key)
    signer_concat = Signer(secret_key=secret, salt=salt, key_derivation="concat", digest_method=digest_method)
    expected_concat = hashlib.sha256(salt + secret).digest()
    assert signer_concat.derive_key() == expected_concat
    assert signer_concat.derive_key(b"other") == hashlib.sha256(salt + b"other").digest()

    # Test "django-concat" derivation: hash(salt + b"signer" + key)
    signer_django = Signer(secret_key=secret, salt=salt, key_derivation="django-concat", digest_method=digest_method)
    expected_django = hashlib.sha256(salt + b"signer" + secret).digest()
    assert signer_django.derive_key() == expected_django
    assert signer_django.derive_key(b"other") == hashlib.sha256(salt + b"signer" + b"other").digest()

    # Test "hmac" derivation: hmac(key, salt)
    signer_hmac = Signer(secret_key=secret, salt=salt, key_derivation="hmac", digest_method=digest_method)
    expected_hmac = hmac.new(secret, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key() == expected_hmac
    
    other_key = b"other-key"
    expected_hmac_other = hmac.new(other_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key(other_key) == expected_hmac_other

    # Test key rotation (list of keys) - uses the last key for default derivation
    keys = [b"old-key", b"new-key"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rotation.derive_key() == b"new-key"
    # Verify it can still derive from the old key if passed explicitly
    assert signer_rotation.derive_key(b"old-key") == b"old-key"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

    # Test string input conversion
    signer_str = Signer(secret_key="string-key", salt=salt, key_derivation="none")
    assert signer_str.derive_key() == b"string-key"
```


# LLM-generated content at query #9
#--------------------------

```python
import pytest
import hmac
import hashlib

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest = hashlib.sha1

    # Test 'concat' derivation
    signer_concat = Signer(secret_key=secret_key, salt=salt, key_derivation="concat", digest_method=digest)
    expected_concat = digest(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(secret/key=secret_key, salt=salt, key_derivation="django-concat", digest_method=digest)
    expected_django = digest(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(secret_key=secret_key, salt=salt, key_derivation="hmac", digest_method=digest)
    mac = hmac.new(secret_key, msg=salt, digestmod=digest)
    expected_hmac = mac.digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(secret_key=secret_key, salt=salt, key_derivation="none")
    assert signer_none.derive_key() == secret_key

    # Test passing specific secret_key to derive_key (rotation support)
    old_key = b"old_secret"
    signer_rotation = Signer(secret_key=[old_key, secret_key], salt=salt, key_derivation="none")
    assert signer_rotation.derive_key(old_key) == old_key

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #10
#--------------------------

```python
import pytest
import hmac
import hashlib

def test_Signer_verify_signature():
    secret_key = b"secret"
    salt = b"salt"
    sep = b"."
    value = b"payload"
    
    # 1. Test with valid signature
    signer = Signer(secret_key=secret_key, salt=salt, sep=sep)
    signature = signer.get_signature(value)
    signed_value = value + sep + signature
    # We extract the signature part from the signed string to test verify_signature directly
    sig_part = signed_value.split(sep)[1]
    assert signer.verify_signature(value, sig_part) is True

    # 2. Test with invalid signature (tampered payload)
    tampered_value = b"tampered_payload"
    assert signer.verify_signature(tampered_value, sig_part) is False

    # 3. Test with incorrect signature
    wrong_sig = b"bm90X2FfcmVhbF9zaWduYXR1cmU=" # base64 for "not_a_real_signature"
    assert signer.verify_signature(value, wrong_sig) is False

    # 4. Test with key rotation (old key signature should still verify)
    old_key = b"old_secret"
    signer_rotation = Signer(secret_key=[old_key, secret_key], salt=salt, sep=sep)
    old_signature = signer_rotation.get_signature(value)
    old_sig_part = (value + sep + old_signature).split(sep)[1]
    assert signer_rotation.verify_signature(value, old_sig_part) is True

    # 5. Test with malformed base64 signature
    assert signer_rotation.verify_signature(value, b"!!!notbase64!!!") is False

    # 6. Test with different derivation method (hmac)
    signer_hmac = Signer(secret_key=secret_key, salt=salt, sep=sep, key_derivation="hmac")
    sig_hmac = signer_hmac.get_signature(value)
    assert signer_hmac.verify_signature(value, sig_hmac) is True

    # 7. Test with NoneAlgorithm (effectively no signature check/empty signature)
    none_algo = NoneAlgorithm()
    signer_none = Signer(secret_key=secret_key, algorithm=none_algo)
    # get_signature returns empty b"", base64 encoded is b""
    assert signer_none.verify_signature(value, b"") is True
```


# LLM-generated content at query #11
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = hashlib.sha256(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = hashlib.sha256(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    expected_hmac = hmac.new(secret_key, digestmod=digest_method)
    expected_hmac.update(salt)
    assert signer_hmac.derive_key() == expected_hmac.digest()

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none"
    )
    assert signer_none.derive_key() == secret_key

    # Test with specific secret_key passed to derive_key
    alt_key = b"alternative"
    signer_alt = Signer(secret_key=secret_key, salt=salt, key_derivation="concat", digest_method=digest_method)
    expected_alt = hashlib.sha256(salt + alt_key).digest()
    assert signer_alt.derive_key(secret_key=alt_key) == expected_alt

    # Test error for unknown derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid_method")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

    # Test rotation: derive_key uses the specific key provided in list if specified
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    # Default should be the last one (new)
    assert signer_rotation.derive_key() == b"new"
    # Explicitly providing 'old'
    assert signer_rotation.derive_key(secret_key=b"old") == b"old"
```


# LLM-generated content at query #12
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(secret_key=secret, salt=salt, key_derivation="concat", digest_mode=digest_method)
    expected_concat = hashlib.sha256(salt + secret).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(secret_key=secret, salt=salt, key_derivation="django-concat", digest_mode=digest_method)
    expected_django = hashlib.sha256(salt + b"signer" + secret).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(secret_key=secret, salt=salt, key_derivation="hmac", digest_mode=digest_method)
    mac = hmac.new(secret, digestmod=digest_method)
    mac.update(salt)
    expected_hmac = mac.digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(secret_key=secret, salt=salt, key_derivation="none")
    assert signer_none.derive_key() == secret

    # Test passing specific secret_key to derive_key
    other_secret = b"other"
    signer_concat_alt = Signer(secret_key=secret, salt=salt, key_derivation="concat", digest_mode=digest_method)
    expected_alt = hashlib.sha256(salt + other_secret).digest()
    assert signer_concat_alt.derive_key(other_secret) == expected_alt

    # Test error for unknown derivation method
    signer_invalid = Signer(secret_key=secret, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

    # Test rotation: derive_key uses newest key by default but can use old ones via parameter
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rotation.derive_key() == b"new"
    assert signer_rotation.derive_key(b"old") == b"old"
```


# LLM-generated content at query #13
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret = b"secret"
    salt = b"salt"
    digest = hashlib.sha1

    # Test 'concat' derivation
    signer_concat = Signer(secret_key=secret, salt=salt, key_derivation="concat", digest_method=digest)
    expected_concat = digest(salt + secret).digest()
    assert signer_canary_derive_key(signer_concat) == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(secret_key=secret, salt=salt, key_derivation="django-concat", digest_method=digest)
    expected_django = digest(salt + b"signer" + secret).digest()
    assert signer_canary_derive_key(signer_django) == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(secret_key=secret, salt=salt, key_derivation="hmac", digest_method=digest)
    mac = hmac.new(secret, digestmod=digest)
    mac.update(salt)
    expected_hmac = mac.digest()
    assert signer_canary_derive_key(signer_hmac) == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(secret_key=secret, salt=salt, key_derivation="none")
    assert signer_canary_derive_key(signer_none) == secret

    # Test with explicit secret_key passed to derive_key
    alt_secret = b"alt-secret"
    signer_concat_alt = Signer(secret_key=secret, salt=salt, key_derivation="concat", digest_method=digest)
    expected_alt = digest(salt + alt_secret).digest()
    assert signer_canary_derive_key(signer_concat_alt, secret_key=alt_secret) == expected_alt

    # Test with rotation (using list of keys)
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    # derive_key without arg should use the newest key (last in list)
    assert signer_canary_derive_key(signer_rotation) == b"new"
    # verify it can derive from the old key if provided
    assert signer_canary_derive_key(signer_rotation, secret_key=b"old") == b"old"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret, key_derivation="invalid_method")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

def signer_canary_derive_key(signer, secret_key=None):
    """Helper to call the method for testing."""
    return signer.derive_key(secret_key)
```


# LLM-generated content at query #14
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"super-secret"
    salt = b"test-salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = digest_method(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = digest_method(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    expected_hmac = hmac.new(secret_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key

    # Test passing specific secret_key to derive_key
    other_key = b"other-key"
    assert signer_concat.derive_key(secret_key=other_key) == digest_method(salt + other_key).digest()

    # Test key rotation (using list of keys)
    keys = [b"old-key", b"new-key"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    # derive_key uses the last key in the list by default
    assert signer_rotation.derive_key() == b"new-key"
    # Manually specifying an older key from the rotation list
    assert signer_rotation.derive_key(secret_key=b"old-key") == b"old-key"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #15
#--------------------------

```python
import pytest

def test_Signer_verify_signature():
    secret_key = b"secret"
    salt = b"salt"
    signer = Signer(secret_key=secret_key, salt=salt)
    value = b"hello world"
    
    # Generate a valid signature
    signature = signer.get_signature(value)
    signed_value = value + signer.sep + signature
    
    # 1. Test valid signature
    assert signer.verify_signature(value, signature) is True
    assert signer.verify_signature(value, signed_value) is True

    # 2. Test invalid signature (tampered content)
    tampered_value = b"tampered world"
    assert signer.verify_signature(tampered_value, signature) is False

    # 3. Test invalid signature (wrong signature for valid content)
    wrong_signature = signer.get_signature(b"something else")
    assert signer.verify_signature(value, wrong_signature) is False

    # 4. Test key rotation: verify with an older key in the list
    old_key = b"old-secret"
    rotation_signer = Signer(secret_key=[old_key, secret_key], salt=salt)
    
    # Signature created with the NEWEST key (secret_key)
    new_sig = rotation_signer.get_signature(value)
    # Verify should succeed because it iterates through keys
    assert rotation_signer.verify_signature(value, new_sig) is True

    # 5. Test signature created with the OLD key
    old_sig_raw = rotation_signer.algorithm.get_signature(rotation_signer.derive_key(old_key), value)
    old_sig_encoded = base64_encode(old_sig_raw)
    assert rotation_signer.verify_signature(value, old_sig_encoded) is True

    # 6. Test malformed base64 signature
    assert signer.verify_signature(value, b"!!!not-base64!!!") is False

    # 7. Test empty/different value
    assert signer.verify_signature(b"", signature) is False
```


# LLM-generated content at query #16
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = hashlib.sha256(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = hashlib.sha256(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    expected_hmac = hmac.new(secret_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key

    # Test passing a specific secret_key to derive_key
    alt_key = b"alternative"
    signer_django.derive_key(secret_key=alt_key)
    expected_alt = hashlib.sha256(salt + b"signer" + alt_key).digest()
    assert signer_django.derive_key(secret_key=alt_key) == expected_alt

    # Test key rotation (list of keys) - should use the last key by default
    keys = [b"old", b"new"]
    signer_rotation = Signer(
        secret_key=keys, 
        salt=salt, 
        key_derivation="none"
    )
    # Default is the newest (last) key
    assert signer_rotation.derive_key() == b"new"
    # Explicitly providing an older key from the list
    assert signer_rotation.derive_key(secret_key=b"old") == b"old"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #17
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key,
        salt=salt,
        key_derivation="concat",
        digest_method=digest_method
    )
    expected_concat = hashlib.sha256(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_cntact
    assert signer_concat.derive_key(b"other") == hashlib.sha256(salt + b"other").digest()

    # Test 'django-concat' derivation
    signer_django = Signer(
        secret_key=secret_key,
        salt=salt,
        key_derivation="django-concat",
        digest_method=digest_method
    )
    expected_django = hashlib.sha256(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django
    assert signer_django.derive_key(b"other") == hashlib.sha256(salt + b"signer" + b"other").digest()

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key,
        salt=salt,
        key_derivation="hmac",
        digest_method=digest_method
    )
    mac = hmac.new(secret_key, digestmod=digest_method)
    mac.update(salt)
    expected_hmac = mac.digest()
    assert signer_hmac.derive_key() == expected_hmac
    
    mac_other = hmac.new(b"other", digestmod=digest_method)
    mac_other.update(salt)
    assert signer_hmac.derive_key(b"other") == mac_other.digest()

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key,
        salt=salt,
        key_derivation="none",
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key
    assert signer_none.derive_key(b"other") == b"other"

    # Test error for unknown derivation
    signer_invalid = Signer(
        secret_key=secret_key,
        salt=salt,
        key_derivation="invalid_method",
        digest_method=digest_method
    )
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

    # Test key rotation (using list of keys)
    keys = [b"old_key", b"new_key"]
    signer_rotation = Signer(
        secret_key=keys,
        salt=salt,
        key_derivation="none"
    )
    # Should default to the newest (last) key
    assert signer_rotation.derive_key() == b"new_key"
    # Providing specific key manually
    assert signer_rotation.derive_key(b"old_key") == b"old_key"
```


# LLM-generated content at query #18
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Signer_verify_signature():
    """
    Tests the verify_signature method of the Signer class, covering:
    - Successful verification with correct signature.
    - Failed verification with incorrect signature.
    - Failed verification with invalid base64 encoding.
    - Verification using rotated keys (oldest to newest).
    """
    secret_key = b"secret"
    salt = b"salt"
    sep = b"."
    value = b"payload"
    
    # Create a signer
    signer = Signer(secret_key=secret_key, salt=salt, sep=sep)
    
    # Generate a valid signature for the value
    # Note: get_signature returns base64 encoded bytes
    valid_sig_encoded = signer.get_signature(value)
    
    # 1. Test successful verification
    assert signer.verify_signature(value, valid_sig_encoded) is True
    
    # 2. Test failed verification with wrong value
    wrong_value = b"wrong_payload"
    assert signer.verify_signature(wrong_value, valid_sig_encoded) is False
    
    # 3. Test failed verification with wrong signature
    wrong_sig_encoded = signer.get_signature(b"another_value")
    assert signer.verify_signature(value, wrong_sig_encoded) is False
    
    # 4. Test failed verification with invalid base64 string
    invalid_base64 = b"!!!not-base64!!!"
    assert signer.verify_signature(value, invalid_base64) is False

    # 5. Test key rotation support
    # We provide a list of keys: [old_key, new_key]. 
    # The signer signs with the newest (last) key.
    # verify_signature should succeed if any key in the list matches.
    old_key = b"old_secret"
    new_key = b"new_secret"
    rotation_signer = Signer(secret_key=[old_key, new_key], salt=salt, sep=sep)
    
    # Create signature using the NEW key (the one used by sign/get_signature)
    sig_with_new_key = rotation_signer.get_signature(value)
    assert rotation_signer.verify_signature(value, sig_with_new_key) is True
    
    # Create signature manually using the OLD key to test rotation verification
    # We must derive the key correctly for the old key
    old_derived_key = rotation_signer.derive_key(old_key)
    # Manually calculate HMAC-SHA1 (default) for the old key
    import hmac, hashlib
    raw_sig = hmac.new(old_derived_key, msg=value, digestmod=hashlib.sha1).digest()
    from .encoding import base64_encode
    sig_with_old_key_encoded = base64_encode(raw_sig)
    
    # verify_signature should find the match when iterating through secret_keys
    assert rotation_signer.verify_signature(value, sig_with_old_key_encoded) is True

    # 6. Test with a custom algorithm (NoneAlgorithm)
    none_algo = NoneAlgorithm()
    algo_signer = Signer(secret_key=secret_key, algorithm=none_algo)
    # NoneAlgorithm returns b"", so base64 of b"" is b""
    assert algo_signer.verify_signature(value, b"") is True
```


# LLM-generated content at query #19
#--------------------------

```python
import pytest
from unittest.mock import MagicMock

def test_Signer_verify_signature():
    # Setup common components
    secret_key = b"secret"
    salt = b"salt"
    value = b"payload"
    sep = b"."
    signer = Signer(secret_key=secret_key, salt=salt, sep=sep)

    # 1. Test successful verification with correct signature
    sig_bytes = signer.get_signature(value)
    assert signer.verify_signature(value, sig_bytes) is True

    # 2. Test failure with incorrect value
    wrong_value = b"wrong_payload"
    assert signer.verify_signature(wrong: wrong_value, sig_bytes) is False

    # 3. Test failure with incorrect signature
    wrong_sig = b"invalid_signature_base64"
    assert signer.verify_signature(value, wrong_sig) is False

    # 4. Test key rotation (verifying with an older key in the list)
    old_key = b"old_secret"
    signer_with_rotation = Signer(secret_key=[old_key, secret_key], salt=salt, sep=sep)
    
    # Generate signature using the old key manually
    old_derived_key = signer_with_rotation.derive_key(old_key)
    old_sig = base64_encode(signer_with_rotation.algorithm.get_signature(old_derived_key, value))
    
    # Should verify True because old_key is in the rotation list
    assert signer_with_rotation.verify_signature(value, old_sig) is True

    # 5. Test failure with a key not in the rotation list
    unrelated_signer = Signer(secret_key=b"unrelated", salt=salt, sep=sep)
    assert unrelated_signer.verify_signature(value, sig_bytes) is False

    # 6. Test failure with malformed base64 input (should return False via try-except)
    malformed_sig = b"!!!" # Not valid base64 in many contexts or contains invalid chars
    assert signer.verify_signature(value, malformed_sig) is False

    # 7. Test using string inputs (ensuring want_bytes handles it)
    str_value = "string_payload"
    str_sig = signer.get_signature(str_value)
    assert signer.verify_signature(str_value, str_sig) is True

    # 8. Test with custom algorithm (Mocking verify_signature)
    mock_algo = MagicMock(spec=SigningAlgorithm)
    mock_algo.verify_signature.return_value = True
    signer_mock = Signer(secret_key=secret_key, algorithm=mock_algo)
    
    assert signer_mock.verify_signature(value, sig_bytes) is True
    mock_algo.verify_signature.assert_called()
```


# LLM-generated content at query #20
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    
    # Test 'concat' derivation
    signer_concat = Signer(secret_key=secret_key, salt=salt, key_derivation="concat")
    expected_concat = hashlib.sha1(salt + secret_key).digest()
    assert signer_can_derive_key_logic(signer_concat, None) == expected_concat

    # Test 'django-concat' derivation
    signer_django = Signer(secret_key=secret_key, salt=salt, key_derivation="django-concat")
    expected_django = hashlib.sha1(salt + b"signer" + secret_key).digest()
    assert signer_can_derive_key_logic(signer_django, None) == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(secret_key=secret_key, salt=salt, key_derivation="hmac")
    expected_hmac = hmac.new(secret_key, msg=salt, digestmod=hashlib.sha1).digest()
    assert signer_can_derive_key_logic(signer_hmac, None) == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(secret_key=secret_key, salt=salt, key_derivation="none")
    assert signer_can_derive_key_logic(signer_none, None) == secret_key

    # Test providing an explicit secret_key to derive_key
    other_key = b"other"
    assert signer_can_derive_key_logic(signer_concat, other_key) == hashlib.sha1(salt + other_key).digest()
    assert signer_can_derive_key_logic(signer_none, other_key) == other_key

    # Test unknown derivation method raises TypeError
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

def signer_can_derive_key_logic(signer, explicit_key):
    """Helper to call the method under test."""
    return signer.derive_key(explicit_key)
```


# LLM-generated content at query #21
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = digest_method(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_contra_expected := expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = digest_method(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    mac = hmac.new(secret_key, digestmod=digest_method)
    mac.update(salt)
    expected_hmac = mac.digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key

    # Test passing an explicit secret_key to derive_key
    alt_key = b"alternative"
    signer_hmac_alt = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    mac_alt = hmac.new(alt_key, digestmod=digest_method)
    mac_alt.update(salt)
    assert signer_hmac_alt.derive_key(secret_key=alt_key) == mac_alt.digest()

    # Test key rotation (using list of keys)
    keys = [b"old_key", b"new_key"]
    signer_rotation = Signer(
        secret_key=keys, 
        salt=salt, 
        key_derivation="none"
    )
    # derive_key() uses the newest (last) key by default
    assert signer_rotation.derive_key() == b"new_key"
    # Manually specifying an older key from the rotation list
    assert signer_rotation.derive_key(secret_key=b"old_key") == b"old_key"

    # Test invalid derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #22
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = digest_method(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_canv_concat := expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = digest_method(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    mac = hmac.new(secret_key, digestmod=digest_method)
    mac.update(salt)
    expected_hmac = mac.digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key

    # Test passing specific secret_key to derive_key
    other_key = b"other"
    assert signer_django.derive_key(secret_key=other_key) == \
           digest_method(salt + b"signer" + other_key).digest()

    # Test rotation support (using the newest key in list)
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    assert signer_rotation.derive_key() == b"new"
    assert signer_rotation.derive_key(secret_key=b"old") == b"old"

    # Test error for unknown derivation
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()
```


# LLM-generated content at query #23
#--------------------------

```python
import pytest
import hashlib
import hmac

def test_Signer_derive_key():
    secret_key = b"secret"
    salt = b"salt"
    digest_method = hashlib.sha256

    # Test 'concat' derivation
    signer_concat = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="concat", 
        digest_method=digest_method
    )
    expected_concat = hashlib.sha256(salt + secret_key).digest()
    assert signer_concat.derive_key() == expected_concat

    # Test 'django-concat' derivation (default)
    signer_django = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="django-concat", 
        digest_method=digest_method
    )
    expected_django = hashlib.sha256(salt + b"signer" + secret_key).digest()
    assert signer_django.derive_key() == expected_django

    # Test 'hmac' derivation
    signer_hmac = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="hmac", 
        digest_method=digest_method
    )
    expected_hmac = hmac.new(secret_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key() == expected_hmac

    # Test 'none' derivation
    signer_none = Signer(
        secret_key=secret_key, 
        salt=salt, 
        key_derivation="none", 
        digest_method=digest_method
    )
    assert signer_none.derive_key() == secret_key

    # Test deriving with an explicit specific key (overriding the default)
    alt_key = b"alternative"
    signer_hmac.derive_key(secret_key=alt_key)
    expected_alt_hmac = hmac.new(alt_key, msg=salt, digestmod=digest_method).digest()
    assert signer_hmac.derive_key(secret_key=alt_key) == expected_alt_hmac

    # Test 'none' derivation with explicit key
    assert signer_none.derive_key(secret_key=alt_key) == alt_key

    # Test error for unknown derivation method
    signer_invalid = Signer(secret_key=secret_key, key_derivation="invalid")
    with pytest.raises(TypeError, match="Unknown key derivation method"):
        signer_invalid.derive_key()

    # Test behavior with list of keys (rotation)
    keys = [b"old", b"new"]
    signer_rotation = Signer(secret_key=keys, salt=salt, key_derivation="none")
    # Should use the newest (last) key by default
    assert signer_rotation.derive_key() == b"new"
    # Explicitly passing an old key
    assert signer_rotation.derive_key(secret_key=b"old") == b"old"
```


