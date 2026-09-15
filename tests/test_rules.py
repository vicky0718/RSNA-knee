"""Regression tests for the rule reader.

Every case here is a bug that shipped once and cost real agreement against the
gold studies, or a language whose absence silently dropped whole institutions.
"""

import pytest

from knee import rules
from knee.reports import LabelState

POSITIVE = LabelState.POSITIVE.value
NEGATIVE = LabelState.NEGATIVE.value
SILENT = LabelState.UNMENTIONED.value


def state(text: str, target: str) -> str:
    return rules.read_report(text)[target]["state"]


class TestNegationBoundaries:
    def test_negation_cue_does_not_match_inside_a_word(self):
        """A bare `no` matched inside "sy-no-vitis" and negated the effusion beside it."""
        text = "Synovitis of left knee and massive joint effusion and suprapatellar bursitis."
        assert state(text, "Effusion") == POSITIVE
        assert state(text, "Synovitis") == POSITIVE

    def test_real_negation_still_negates(self):
        assert state("There is no joint effusion.", "Effusion") == NEGATIVE
        assert state("Kein Gelenkerguss.", "Effusion") == NEGATIVE
        assert state("Geen gewrichtsvocht.", "Effusion") == NEGATIVE


class TestStructuresVersusEntities:
    def test_naming_a_structure_is_not_a_positive(self):
        assert state("The ACL is normal. The MCL is intact.", "ACL") == NEGATIVE

    def test_structure_needs_pathology_for_a_positive(self):
        assert state("Complete tear of the anterior cruciate ligament.", "ACL") == POSITIVE

    def test_entity_mention_is_the_finding(self):
        assert state("Small Baker's cyst.", "Baker's") == POSITIVE


class TestCartilageIsAnatomyNotPathology:
    def test_healthy_cartilage_is_not_osteoarthritis(self):
        """`cartilag\\w*` sat in the OA pathology list, so any cartilage mention read positive."""
        text = "Cartilagos femorotibiales y patelofemorales sin alteraciones."
        assert state(text, "PF OA") == NEGATIVE

    def test_negation_is_checked_at_every_pathology_word(self):
        """Only the first match was tested, so an earlier un-negated word won."""
        text = "The patellar and trochlear cartilage appears congruent without chondromalacia."
        assert state(text, "PF OA") == NEGATIVE

    def test_cartilage_damage_is_osteoarthritis(self):
        assert state("Condropatia rotuliana grado III.", "PF OA") == POSITIVE


class TestPatellaIsNotOnlyCartilage:
    def test_patellar_tendon_is_not_a_patellofemoral_finding(self):
        assert state("The quadriceps and patellar tendons are preserved.", "PF OA") == SILENT


class TestMeniscusLabelIsATear:
    def test_degeneration_short_of_a_tear_is_not_a_tear(self):
        text = "Grade II degenerative signal in the medial meniscus without a tear."
        assert state(text, "Medial Meniscus") == NEGATIVE

    def test_tear_is_a_tear(self):
        assert state("Medial meniscus posterior horn tear.", "Medial Meniscus") == POSITIVE

    def test_words_may_sit_between_the_side_and_the_structure(self):
        text = "There is a tear of the posterior horn of the medial meniscus."
        assert state(text, "Medial Meniscus") == POSITIVE

    def test_sides_are_not_confused(self):
        text = "Lateral meniscus tear. Medial meniscus normal."
        assert state(text, "Lateral Meniscus") == POSITIVE
        assert state(text, "Medial Meniscus") == NEGATIVE


class TestLanguageCoverage:
    """A monolingual reader does not mislabel a study, it drops it — invisibly."""

    @pytest.mark.parametrize(
        "text,target,expected",
        [
            ("Στο εσω μηνισκο παρατηρειται ρηξη.", "Medial Meniscus", POSITIVE),
            ("Ο προσθιος χιαστος συνδεσμος ειναι φυσιολογικος.", "ACL", NEGATIVE),
            ("Diz eklemi içi sıvı miktarı hafif derecede artmış.", "Effusion", POSITIVE),
            ("Ön çapraz bağ rüptürü mevcut.", "ACL", POSITIVE),
            ("Rotura del menisco interno.", "Medial Meniscus", POSITIVE),
            ("Riss des vorderen Kreuzbandes.", "ACL", POSITIVE),
            ("Voorste kruisband intact.", "ACL", NEGATIVE),
            ("Разрыв передней крестообразной связки.", "ACL", POSITIVE),
            ("Il n'y a pas d'épanchement significatif.", "Effusion", NEGATIVE),
        ],
    )
    def test_reads_the_corpus_languages(self, text, target, expected):
        assert state(text, target) == expected


def test_silence_stays_silence():
    """The commonest failure mode in this task is scoring silence as a negative."""
    assert state("MRI of the knee. No acute findings.", "Baker's") == SILENT
    assert state("MRI of the knee.", "ACL") == SILENT
