from dataclasses import dataclass

GRAVE_ACCENT = "\u0300"
ACUTE_ACCENT = "\u0301"
TILDE_ACCENT = "\u0303"
STRESS_MARKS = GRAVE_ACCENT + ACUTE_ACCENT + TILDE_ACCENT
STRESS_LETTERS = "aąeęėiįylmnoruųū"


@dataclass(frozen=True)
class Symbol:
    id: int
    token: str


PAD = Symbol(id=0, token="*")
UNK = Symbol(id=1, token="#")
SOS = Symbol(id=2, token="<")
EOS = Symbol(id=3, token=">")
