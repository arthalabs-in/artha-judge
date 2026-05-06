from __future__ import annotations

import re


DISPOSITION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("partly_allowed", re.compile(r"\b(partly|partially)\s+allowed\b", re.I)),
    ("allowed", re.compile(r"\b(?:petition|appeal|application|writ petition)\s+(?:is\s+)?allowed\b|\ballowed\b", re.I)),
    ("dismissed", re.compile(r"\b(?:petition|appeal|application)\s+(?:is\s+)?dismissed\b|\bdismissed\b", re.I)),
    ("disposed", re.compile(r"\b(?:petition|appeal|matter)\s+(?:is\s+)?disposed\b|\bdisposed\s+of\b", re.I)),
    ("quashed", re.compile(r"\bquash(?:ed|ing)?\b", re.I)),
    ("remanded", re.compile(r"\bremand(?:ed)?\b", re.I)),
]

LEGAL_PHRASE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("timeline", re.compile(r"\bwithin\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+(?:days?|weeks?|months?)\b", re.I)),
    ("liberty_to_appeal", re.compile(r"\bliberty\s+to\s+(?:appeal|approach|file)\b", re.I)),
    ("compliance_report", re.compile(r"\bcompliance\s+report\b", re.I)),
    ("affidavit", re.compile(r"\baffidavit\b", re.I)),
    ("speaking_order", re.compile(r"\bspeaking\s+order\b", re.I)),
    ("reconsideration", re.compile(r"\breconsider(?:ation|ed)?\b", re.I)),
    ("payment_release", re.compile(r"\b(?:release|pay|payment|arrears|benefits?|compensation)\b", re.I)),
    ("records_update", re.compile(r"\b(?:update|correct|modify)\s+(?:records?|entries|register)\b", re.I)),
    ("appeal", re.compile(r"\b(?:appeal|appellate|special leave)\b", re.I)),
]

DEPARTMENT_PATTERN = re.compile(
    r"\b(?:Government of [A-Z][A-Za-z ]+|State of [A-Z][A-Za-z ]+|[A-Z][A-Za-z& ]+ "
    r"(?:Department|Corporation|Authority|Board|Commissioner|Ministry|Police|Revenue|Municipality|Panchayat)|"
    r"BBMP|BDA|PWD|KSRTC|Devaswom Board|Travancore Devaswom Board)\b",
    re.I,
)

OPERATIVE_MARKER_RE = re.compile(
    r"\b(?:we\s+(?:direct|order|allow|dismiss|dispose)|"
    r"(?:is|are)\s+directed\s+to|ordered\s+to|shall\s+(?:file|pay|release|reconsider|issue|remove|update|comply)|"
    r"writ\s+petition\s+is\s+(?:allowed|dismissed|disposed)|petition\s+is\s+(?:allowed|dismissed|disposed))\b",
    re.I,
)

NON_OPERATIVE_RE = re.compile(
    r"\b(?:petitioner(?:s)?\s+(?:submit|contend|refer|cite|aver|asseverate)|"
    r"respondent(?:s)?\s+(?:submit|contend|refer|cite)|"
    r"high\s+court\s+issued\s+the\s+following\s+directions|"
    r"section\s+\d+|article\s+\d+|stipulates|provides|states|counsel\s+submitted|"
    r"has\s+submitted|have\s+submitted|has\s+referred|have\s+referred)\b",
    re.I,
)
