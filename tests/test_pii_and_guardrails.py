import pytest

from coverwise.guardrails import classify_intent
from coverwise.pii import mask_pii


@pytest.mark.parametrize("text,label", [
    ("email me at jane.doe@example.com", "EMAIL"),
    ("my Medicare number is 1EG4-TE5-MK73", "MEDICARE_ID"),
    ("SSN 123-45-6789", "SSN"),
    ("born 04/12/1951", "DATE"),
    ("call (555) 123-4567", "PHONE"),
    ("member ID: ABC123456", "MEMBER_ID"),
])
def test_masks_each_pii_type(text, label):
    masked, found = mask_pii(text)
    assert f"[{label}]" in masked
    assert found[label] == 1


def test_leaves_clean_text_alone():
    text = "Does Medicare Part B cover X-rays?"
    assert mask_pii(text) == (text, {})


@pytest.mark.parametrize("question,intent", [
    ("Does Medicare cover hearing aids?", "coverage"),
    ("How much do I pay for an X-ray under Part B?", "coverage"),
    ("Should I take ibuprofen for my knee pain?", "medical_advice"),
    ("What dosage of insulin is right for me?", "medical_advice"),
    ("Who won the cricket world cup?", "off_topic"),
    ("Write me a poem about the sea", "off_topic"),
])
def test_intent(question, intent):
    assert classify_intent(question) == intent
