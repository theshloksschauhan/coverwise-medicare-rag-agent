"""Input guardrail: decide whether a (masked) question is in scope before any retrieval or LLM call."""
import re

from coverwise.schemas import Intent

MEDICAL_ADVICE = re.compile(
    r"\b(should i (take|stop|start|get|use)|is it safe|safe (for me )?to|dos(e|age)|how much .* should i"
    r"|diagnos(e|is)|what('s| is) wrong with me|treat(ment)? for my|cure|side effects? of|prescribe me"
    r"|which (medicine|medication|drug) should|do i have)\b",
    re.IGNORECASE,
)

DOMAIN = re.compile(
    r"\b(medicare|medicaid|cover(ed|age|s)?|pay(s|ment)?|cost|copay(ment)?|coinsurance|deductible|part [abcd]"
    r"|plan|insurance|benefit|claim|prior auth\w*|referral|eligib\w*|approved amount|out[- ]of[- ]pocket"
    r"|therapy|vaccine|shot|screening|test|scan|x-?ray|mri|surgery|hospice|nursing|ambulance|dialysis|dental"
    r"|hearing|wheelchair|scooter|oxygen|telehealth|prescription|drug|hospital|doctor|equipment|care)\b",
    re.IGNORECASE,
)

MEDICAL_ADVICE_REFUSAL = (
    "I can't give medical advice. Please ask your doctor or pharmacist. "
    "I can answer what Medicare covers and what you may pay."
)
OFF_TOPIC_REFUSAL = "I only answer questions about Medicare coverage, such as what is covered and what it costs."


def classify_intent(question: str) -> Intent:
    if MEDICAL_ADVICE.search(question):
        return "medical_advice"
    if not DOMAIN.search(question):
        return "off_topic"
    return "coverage"


def refusal_for(intent: Intent) -> str:
    return MEDICAL_ADVICE_REFUSAL if intent == "medical_advice" else OFF_TOPIC_REFUSAL
