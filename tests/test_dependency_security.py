"""Check the actual trust bundle used by certifi-dependent clients."""

import ssl

import certifi


def test_certifi_bundle_excludes_compromised_emsign_trust_root() -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=certifi.where())
    authorities = context.get_ca_certs()

    # Legitimate clients must still receive a usable CA store.
    assert authorities
    common_names = {
        value
        for authority in authorities
        for relative_name in authority["subject"]
        for key, value in relative_name
        if key == "commonName"
    }
    assert "emSign ECC Root CA - G3" not in common_names
