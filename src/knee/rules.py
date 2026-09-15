"""A multilingual rule reader for the twelve findings.

Reports arrive in roughly ten languages — English, Turkish, Dutch, Spanish,
Russian, German, Polish, French, Croatian and more — and a monolingual reader
silently drops whole institutions. A dropped study is not a wrong label; it is
no label at all, which is worse, because it is invisible.

Two kinds of finding need different logic:

- *Structures* (cruciate and collateral ligaments, menisci, cartilage) are named
  in almost every report, usually to say they are normal. Naming one is not a
  positive; it takes an explicit pathology word nearby.
- *Entities* (effusion, synovitis, Baker's cyst, contusion, fracture) are named
  only when present or explicitly excluded, so the term itself is the finding
  and negation is what has to be detected.

Measured against the 58 gold studies (bin/score_labels.py), this reader scores
0.748 macro AUC once calibrated. The best public LLM table scores 0.893 — which
does reproduce its author's published figure, as AUC rather than as agreement at
a 0.5 threshold. So this module is the no-GPU fallback, not the label source:
blending it into the public table lowered the score at every weight tested.
Where it earns its keep is the three-state reading (`positive`, `negative`,
`not_mentioned`) that `reports.py` calibrates, and as an auditable second
opinion on the findings the public table reads worst.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .constants import TARGETS
from .reports import LabelState

# --- shared vocabulary -------------------------------------------------------


def _alts(*alternatives: str) -> str:
    """Join alternatives with boundaries on both sides.

    Every cue here is short and many are substrings of ordinary clinical words:
    an unanchored `no` matches inside "sy-no-vitis" and turns "massive joint
    effusion" into a negative. Boundaries are not a nicety in this lexicon.
    """
    return r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)"


#: Cues that make a nearby finding word not apply, across the corpus languages.
NEGATION = _alts(
    r"no", r"not", r"without", r"absent", r"negative", r"free\s+of", r"rule[sd]?\s+out",
    r"sin", r"ningun\w*", r"no\s+se\s+(?:observa|identifica|evidencia|aprecia)",
    r"geen", r"niet", r"zonder",
    r"kein\w*", r"nicht", r"ohne",
    r"pas\s+d\w*", r"aucun\w*", r"sans", r"absence",
    r"yok\w*", r"izlenmedi", r"saptanmad\w*", r"gorulmedi", r"mevcut\s+degil",
    r"нет", r"не", r"без", r"отсутству\w*",
    r"brak", r"nie", r"bez",
    r"nema", r"nije",
    r"non", r"senza", r"assenza", r"nessun\w*",
    r"δεν", r"χωρις", r"ουδεμια", r"απουσια", r"ελευθερ\w*",
)

#: Cues asserting a structure is healthy. A structure sentence carrying one of
#: these and no pathology word is an explicit negative, not silence.
NORMAL = _alts(
    r"normal\w*", r"intact\w*", r"unremarkable", r"preserved", r"no\s+abnormal\w*",
    r"within\s+normal\s+limits",
    r"sin\s+(?:alteraciones|anomal\w*|lesion\w*)", r"conservad\w*", r"integr\w*",
    r"ongestoord", r"normaal", r"gaaf",
    r"regelrecht\w*", r"unauffallig\w*", r"intakt\w*", r"erhalten",
    r"preserve\w*", r"conserve\w*", r"sans\s+particularite\w*",
    r"korunmus\w*", r"saglam", r"olagan", r"dogal", r"tabii",
    r"норм\w*", r"сохран\w*", r"цел\w*",
    r"prawidlow\w*", r"zachowan\w*",
    r"uredan", r"uredna", r"ocuvan\w*", r"intaktn\w*",
    r"φυσιολογικ\w*", r"ακεραι\w*", r"αθικτ\w*", r"ελευθερ\w*",
)

_TEAR = _alts(
    r"tears?", r"torn", r"ruptur\w*", r"disrupt\w*", r"discontinu\w*", r"lesions?",
    r"rotur\w*", r"desgarr\w*", r"lesion\w*",
    r"scheur\w*", r"ruptuur", r"letsel",
    r"riss\w*", r"einriss\w*", r"abriss\w*", r"lasion\w*",
    r"dechirur\w*", r"fissurat\w*",
    r"y[ıi]rt[ıi]k\w*", r"rupt[uü]r\w*", r"yaralanma\w*", r"lezyon\w*",
    r"разрыв\w*", r"надрыв\w*", r"повреждени\w*",
    r"uszkodzeni\w*", r"zerwani\w*", r"pekni\w*",
    r"puknu\w*", r"ozljed\w*", r"lezij\w*",
    r"rottur\w*", r"lacerazion\w*",
    r"ρηξη", r"ρηξεις", r"ρωγμ\w*",
)

_DEGEN = _alts(
    r"degenerat\w*", r"degenerativ\w*", r"mucoid", r"myxoid",
    r"дегенерат\w*", r"zwyrodnieni\w*", r"dejeneratif", r"dejenerasyon",
    r"εκφυλιστικ\w*", r"εκφυλισ\w*",
)

_OA = _alts(
    r"osteoarthrit\w*", r"arthros\w*", r"artros\w*", r"arthrose\w*", r"gonarthros\w*",
    r"gonartroz\w*", r"osteofit\w*", r"osteophyt\w*",
    r"chondropath\w*", r"condropat\w*", r"chondromalac\w*", r"condromalac\w*",
    r"chondrose", r"chondrosis", r"chondral\s+(?:defect|loss|thinning|damage|injur\w*|fissur\w*)",
    r"cartilage\s+(?:loss|thinning|defect|damage|wear|fissur\w*|degenerat\w*)",
    r"(?:thinning|loss|defect|fissur\w*|erosion)\s+of\s+the\s+(?:articular\s+)?cartilage",
    r"knorpel(?:schaden|verschmalerung|defekt|lasion)\w*", r"kraakbeen(?:schade|verlies)\w*",
    r"kikirdak\s+(?:kaybi|incelme\w*|hasar\w*|defekt\w*)", r"k[ıi]k[ıi]rdak\s+\w*(?:kayb|incel)\w*",
    r"stanjen\w*\s+hrskavic\w*", r"hrskavic\w*\s+stanjen\w*",
    r"артроз\w*", r"хондромаляц\w*", r"хондропат\w*",
    r"zwyrodnieni\w*\s+chrzastk\w*", r"chondropatia",
    r"χονδροπαθει\w*", r"οστεοαρθρ\w*", r"οστεοφυτ\w*", r"χονδρομαλακ\w*",
)

#: Cartilage as *anatomy*: it belongs on the anatomy side of an OA rule, not the
#: pathology side. "the cartilage is preserved" names cartilage and asserts health.
_CARTILAGE = (
    r"cartilag\w*|cartilago\w*|knorpel\w*|kraakbeen\w*|kikirdak\w*|k[ıi]k[ıi]rdak\w*"
    r"|hrskavic\w*|chrzastk\w*|хрящ\w*|χονδρ\w*|condral|chondral"
)


# --- rule table --------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """How one finding is read out of a sentence."""

    anatomy: str
    #: For structures: the pathology that makes a mention positive.
    #: For entities: empty, because naming the entity is the finding.
    pathology: str = ""

    @property
    def is_entity(self) -> bool:
        return not self.pathology


_MEDIAL = (
    r"medial\w*|intern[oa]s?|interne\w*|inner|innen\w*|mediaal|medyal\w*|mediale\w*"
    r"|przysrodkow\w*|внутренн\w*|medijaln\w*|εσω"
)
_LATERAL = (
    r"lateral\w*|extern[oa]s?|externe\w*|outer|aussen\w*|lateraal|d[ıi]s\w*"
    r"|boczn\w*|наружн\w*|εξω"
)
_MENISCUS = r"meni[sc]\w+|lakotk\w+|мениск\w*|μηνισκ\w*"

#: Up to three words may sit between the side and the structure: reports say
#: "posterior horn of the medial meniscus" as readily as "medial meniscus", and
#: requiring adjacency drops the first form entirely.
_NEAR = r"[\s\-]+(?:\w+[\s\-]+){0,3}"

RULES: dict[str, Rule] = {
    "ACL": Rule(
        r"(?<!\w)acl(?!\w)|anterior\s+cruciate|cruzado\s+anterior|croise\s+anterieur"
        # German declines: "vorderes Kreuzband", "des vorderen Kreuzbandes".
        r"|vorder\w*\s+kreuzband\w*|(?<!\w)vkb(?!\w)|voorste\s+kruisband"
        r"|on\s+capraz|(?<!\w)ocb(?!\w)|przednie\s+wiezadlo\s+krzyzowe"
        r"|передн\w*\s+крестообразн\w*|prednj\w*\s+ukrizen\w*|(?<!\w)lca(?!\w)"
        r"|crociato\s+anteriore|προσθι\w*\s+χιαστ\w*",
        _TEAR,
    ),
    "MCL": Rule(
        r"medial\s+collateral|collateral\s+medial|colateral\s+(?:medial|interno)"
        r"|collateral\s+(?:medial|interne)|innenband|mediales?\s+(?:kollateral|seitenband)"
        r"|mediale\s+collaterale|binnenband|ic\s+yan\s+bag\w*"
        r"|wiezadlo\s+poboczne\s+piszczelowe|внутренн\w*\s+боков\w*"
        r"|medijaln\w*\s+kolateraln\w*|(?<!\w)mcl(?!\w)|(?<!\w)lcm(?!\w)"
        r"|εσω\s+πλαγι\w*",
        _TEAR,
    ),
    # The label is a *tear*. Degeneration is not a tear: a grade II degenerative
    # signal that stops short of the articular surface is routinely reported and
    # routinely not labelled, so `_DEGEN` stays out of the meniscus pathology.
    "Medial Meniscus": Rule(
        rf"(?:{_MEDIAL}){_NEAR}(?:{_MENISCUS})|(?:{_MENISCUS}){_NEAR}(?:{_MEDIAL})",
        _TEAR,
    ),
    "Lateral Meniscus": Rule(
        rf"(?:{_LATERAL}){_NEAR}(?:{_MENISCUS})|(?:{_MENISCUS}){_NEAR}(?:{_LATERAL})",
        _TEAR,
    ),
    "Medial OA": Rule(
        rf"(?:{_MEDIAL})[\s-]+(?:compartment\w*|compartiment\w*|kompartiment\w*|kompartman\w*"
        rf"|femorotibial\w*|tibiofemoral\w*|femoral\s+condyle|tibial\s+plateau|femurkondyl\w*"
        rf"|femoral\s+kondil\w*|tibia\s+platos\w*|condilo\s+femoral|platillo\s+tibial)"
        rf"|(?:compartment|femorotibial\w*|tibiofemoral\w*|condyle|plateau|kondil\w*|plato\w*)"
        rf"[\s-]+(?:{_MEDIAL})",
        _OA,
    ),
    "Lateral OA": Rule(
        rf"(?:{_LATERAL})[\s-]+(?:compartment\w*|compartiment\w*|kompartiment\w*|kompartman\w*"
        rf"|femorotibial\w*|tibiofemoral\w*|femoral\s+condyle|tibial\s+plateau|femurkondyl\w*"
        rf"|femoral\s+kondil\w*|tibia\s+platos\w*|condilo\s+femoral|platillo\s+tibial)"
        rf"|(?:compartment|femorotibial\w*|tibiofemoral\w*|condyle|plateau|kondil\w*|plato\w*)"
        rf"[\s-]+(?:{_LATERAL})",
        _OA,
    ),
    "PF OA": Rule(
        rf"patellofemoral\w*|patelofemoral\w*|patello-femoral\w*|retropatellar\w*"
        r"|femoropatel\w*|patellar\s+cartilage|patellaknorpel|patellofemorale\w*"
        r"|пателлофеморальн\w*|rzepkowo-udow\w*|trochlea\w*|troclea\w*|trohlear\w*"
        r"|rotulian\w*|patellar\s+chondr\w*|patellofemoral\s+eklem"
        # "patella" alone also names the tendon and the retinaculum, neither of
        # which is cartilage; require a cartilage word alongside it.
        rf"|patella\w*[\s-]+(?:{_CARTILAGE})|(?:{_CARTILAGE})[\s-]+(?:de\s+la\s+)?r?otul\w*"
        r"|επιγονατιδομηριαι\w*|επιγονατιδ\w*",
        _OA,
    ),
    "Effusion": Rule(
        r"effusion\w*|derrame\w*|epanchement\w*|erguss|gelenkerguss|gewrichtsvocht"
        r"|vocht\s+in|hydrops|eklem\w*\s+ic\w*\s+s[ıi]v[ıi]\w*|eklem\s+s[ıi]v[ıi]\w*"
        r"|efuzyon|wysiek\w*|выпот\w*|жидкост\w*|izliv\w*|joint\s+fluid"
        r"|fluid\s+in\s+the\s+joint|s[ıi]v[ıi]\s+art[ıi]s\w*|αρθρικ\w*\s+υγρ\w*"
        r"|συλλογ\w*\s+υγρ\w*|υγρο\s+εντος"
    ),
    "Synovitis": Rule(
        r"synovit\w*|sinovit\w*|synovial\s+(?:thicken\w*|proliferat\w*|hypertroph\w*|enhance\w*)"
        r"|synovial\w*\s+(?:verdickung|proliferation)|sinovyal\w*|синовит\w*"
        r"|zapalenie\s+blony\s+maziowej|sinovijaln\w*|pannus|υμενιτιδ\w*|υμενιτιδα"
    ),
    "Baker's": Rule(
        r"baker\w*|popliteal\s+cyst|quiste\s+poplite\w*|kyste\s+poplite\w*"
        r"|poplitealzyste|bakerzyste|popliteale\s+cyste|popliteal\s+kist\w*|baker\s+kist\w*"
        r"|torbiel\s+bakera|киста\s+бейкера|подколенн\w*\s+кист\w*|poplitealn\w*\s+cist\w*"
        r"|cisti\s+di\s+baker|ιγνυακ\w*\s+κυστ\w*|κυστη\s+baker"
    ),
    "Contusion": Rule(
        r"bone\s+(?:bruise|contusion)|contusion\s+osse\w*|contusion\s+ose\w*|knochenkontusion"
        r"|bone\s+marrow\s+(?:o?edema|oedeem)|edema\s+(?:oseo|de\s+medula|osseux|medular)"
        r"|knochen[o]dem|knochenmark[so]dem|beenmerg[o]edeem|kemik\s+(?:iligi\s+)?odem\w*"
        r"|отек\s+костного\s+мозга|obrzek\s+szpiku|kostani\s+edem|kontuzi\w*"
        r"|bone\s+marrow\s+signal|trabecular\s+edema|medullar\w*\s+edema"
        r"|οιδημα\s+μυελου|μυελικ\w*\s+οιδημα"
    ),
    "Fracture": Rule(
        r"fractur\w*|fraktur\w*|frattur\w*|breuk|k[ıi]r[ıi]k\w*|zlamani\w*|перелом\w*"
        r"|prijelom\w*|avulsion\w*|avulsie|abrissfraktur|καταγμα\w*|καταγματ\w*"
    ),
}


_SENTENCE = re.compile(r"[.;:\n]+")
_WINDOW = 45


def _fold(text: str) -> str:
    """Casefold and strip accents, keeping Cyrillic and CJK intact."""
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(_fold(pattern), re.IGNORECASE | re.UNICODE)


_ANATOMY = {t: _compiled(r.anatomy) for t, r in RULES.items()}
_PATHOLOGY = {t: _compiled(r.pathology) for t, r in RULES.items() if r.pathology}
_NEGATION = _compiled(NEGATION)
_NORMAL = _compiled(NORMAL)


def _negated(sentence: str, at: int) -> bool:
    """Is there a negation cue just before this match?"""
    window = sentence[max(0, at - _WINDOW) : at]
    return bool(_NEGATION.search(window))


def read_report(report: str, targets: tuple[str, ...] = TARGETS) -> dict[str, dict]:
    """Read one report into a state, confidence and the sentence used as evidence."""
    text = _fold(report or "")
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    out = {
        t: {"state": LabelState.UNMENTIONED.value, "confidence": 0.0, "evidence": ""}
        for t in targets
    }

    for target in targets:
        rule = RULES.get(target)
        if rule is None:
            continue
        anatomy = _ANATOMY[target]
        best: tuple[str, float, str] | None = None
        for sentence in sentences:
            hit = anatomy.search(sentence)
            if not hit:
                continue
            if rule.is_entity:
                state, confidence = (
                    (LabelState.NEGATIVE.value, 0.8)
                    if _negated(sentence, hit.start()) or _NORMAL.search(sentence)
                    else (LabelState.POSITIVE.value, 0.8)
                )
            else:
                found = list(_PATHOLOGY[target].finditer(sentence))
                pathology = found[0] if found else None
                if any(not _negated(sentence, m.start()) for m in found):
                    state, confidence = LabelState.POSITIVE.value, 0.8
                elif _NORMAL.search(sentence) or pathology:
                    # "normal", or a pathology word that was explicitly negated
                    state, confidence = LabelState.NEGATIVE.value, 0.7
                else:
                    # named, nothing said about it: silence, not a negative
                    state, confidence = LabelState.UNMENTIONED.value, 0.2
            candidate = (state, confidence, sentence[:200])
            # a positive anywhere in the report outranks a negative elsewhere
            if best is None or _priority(state) > _priority(best[0]):
                best = candidate
        if best is not None:
            out[target] = {"state": best[0], "confidence": best[1], "evidence": best[2]}
    return out


def _priority(state: str) -> int:
    return {
        LabelState.POSITIVE.value: 2,
        LabelState.NEGATIVE.value: 1,
        LabelState.UNMENTIONED.value: 0,
    }[state]


def read_corpus(reports, targets: tuple[str, ...] = TARGETS):
    """Read a Series of reports into (states, confidence) frames."""
    import pandas as pd

    rows = [read_report(r, targets) for r in reports]
    states = pd.DataFrame(
        [{t: row[t]["state"] for t in targets} for row in rows], index=reports.index
    )
    confidence = pd.DataFrame(
        [{t: row[t]["confidence"] for t in targets} for row in rows], index=reports.index
    )
    return states, confidence
