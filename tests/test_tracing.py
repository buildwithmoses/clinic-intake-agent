from clinic_intake_agent.tracing import anonymizer_rules, configure_trace_redaction, redact_text


def test_redact_text_masks_phi():
    text = "Riley, DOB 05/09/1979, phone (555) 010-0401, riley@example.com, SSN 900-00-0401, dob 1979-05-09"
    out = redact_text(text)
    for secret in ["05/09/1979", "010-0401", "riley@example.com", "900-00-0401", "1979-05-09"]:
        assert secret not in out
    assert "<ssn>" in out and "<phone>" in out and "<email>" in out and "<date>" in out


def test_anonymizer_rules_are_compiled():
    assert all(hasattr(r["pattern"], "sub") for r in anonymizer_rules())


def test_redaction_opt_out(monkeypatch):
    monkeypatch.setenv("CLINIC_REDACT_TRACES", "false")
    assert configure_trace_redaction() is False


def test_retry_only_transient_errors():
    import anthropic
    import httpx

    from clinic_intake_agent.middleware import is_transient

    req = httpx.Request("POST", "https://example.invalid")
    assert is_transient(anthropic.APITimeoutError(request=req))
    assert is_transient(anthropic.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None))
    assert not is_transient(TypeError("Anthropic authentication failed"))
    assert not is_transient(anthropic.BadRequestError("bad", response=httpx.Response(400, request=req), body=None))
