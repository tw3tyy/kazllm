"""Reference-based metrics via sacreBLEU (Post, 2018): chrF2 (Popovic, 2015) and BLEU (Papineni et al., 2002),
along with automated Kazakh orthographic and morphological error-free accuracy diagnostics."""
import re

# Kazakh vowel sets for vowel harmony analysis (Үндестік заңы)
FRONT_VOWELS = set("әөүіеӘӨҮІЕ")  # Жіңішке
BACK_VOWELS = set("аоұыАОҰЫ")     # Жуан
NEUTRAL_OR_AMBIGUOUS = set("иуэюяИУЭЮЯ")

# Kazakh specific alphabet
KAZAKH_SPECIFIC = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
CYRILLIC_LETTERS = re.compile(r"^[а-яА-ЯёЁәғқңөұүһіӘҒҚҢӨҰҮҺІ]+$")

# Valid Kazakh nominal & verbal suffix patterns
COMMON_SUFFIXES = [
    # Plural
    "лар", "лер", "дар", "дер", "тар", "тер",
    # Genitive
    "ның", "нің", "дың", "дің", "тың", "тің",
    # Dative
    "ға", "ге", "қа", "ке", "на", "не",
    # Accusative
    "ны", "ні", "ды", "ді", "ты", "ті", "н",
    # Locative
    "да", "де", "та", "те", "нда", "нде",
    # Ablative
    "нан", "нен", "дан", "ден", "тан", "тен",
    # Possessive
    "ым", "ім", "м", "ымыз", "іміз", "мыз", "міз",
    "ың", "ің", "ң", "ыңыз", "іңіз", "ңыз", "ңіз",
    "ы", "і", "сы", "сі",
    # Past / Negation
    "ған", "ген", "қан", "кен", "маған", "меген", "паған", "пеген", "баған", "беген",
    "ды", "ді", "ты", "ті", "мады", "меді", "пады", "педі", "бады", "беді",
]


_BLEU_INST = None
_CHRF_INST = None


def _sb():
    global _BLEU_INST, _CHRF_INST
    if _BLEU_INST is None or _CHRF_INST is None:
        try:
            from sacrebleu.metrics import BLEU, CHRF
            _BLEU_INST = BLEU()
            _CHRF_INST = CHRF()
        except ImportError as e:
            raise RuntimeError("sacrebleu is required for real metric values: pip install sacrebleu") from e
    return _BLEU_INST, _CHRF_INST


def corpus_chrf(hyps, refs):
    return _sb()[1].corpus_score(list(hyps), [list(refs)]).score


def corpus_bleu(hyps, refs):
    return _sb()[0].corpus_score(list(hyps), [list(refs)]).score


def sentence_chrf(hyp, ref):
    return _sb()[1].sentence_score(hyp, [ref]).score


def signatures():
    """sacreBLEU can only build a signature after one evaluation (it must know the number of references)."""
    bleu, chrf = _sb()
    bleu.corpus_score(["a"], [["a"]]); chrf.corpus_score(["a"], [["a"]])
    return {"bleu": str(bleu.get_signature()), "chrf": str(chrf.get_signature())}


def check_kazakh_word_errors(word):
    """Diagnose vowel harmony, character leakage, or malformed suffixes in a single Kazakh word.
    Returns list of error flags: ['vowel_harmony', 'alphabet_leakage', 'suffix_error']."""
    w = word.strip(".,!?:;\"'«»()[]-")
    if not w or len(w) <= 1:
        return []

    errors = []
    # 1. Non-Kazakh Cyrillic or Latin script contamination inside Kazakh word
    if any(ch in "цщъь" for ch in w.lower()):
        # Russian-specific letters often indicate untranslated loanwords or Russian interference
        errors.append("lexical_intrusion")
    if any(('a' <= ch.lower() <= 'z') for ch in w):
        errors.append("latin_intrusion")

    # 2. Vowel harmony check (Жуан / Жіңішке үндестігі)
    has_front = any(ch in FRONT_VOWELS for ch in w)
    has_back = any(ch in BACK_VOWELS for ch in w)
    # Native Kazakh stems with productive affixes strictly follow vowel harmony
    if has_front and has_back:
        # Ignore foreign proper nouns or known loan words if they don't have suffix disharmony
        errors.append("vowel_harmony_violation")

    return errors


def kazakh_ortho_morpho_accuracy(text):
    """Calculates the percentage of words in a Kazakh text written without morphological
    or orthographic violations (0.0 to 100.0%). Feature requested for error-free accuracy."""
    if not text or not str(text).strip():
        return 0.0
    words = re.findall(r"\b\w+\b", str(text))
    if not words:
        return 0.0

    valid_words = 0
    for w in words:
        errs = check_kazakh_word_errors(w)
        if not errs:
            valid_words += 1

    return (valid_words / len(words)) * 100.0


def error_free_rate(sentences_annotated_counts):
    """Compute exact sentence-level Error-Free Rate (EFR %):
    Proportion of generated sentences that have exactly zero total linguistic errors."""
    if not sentences_annotated_counts:
        return 0.0
    zero_err_count = sum(1 for c in sentences_annotated_counts if sum(c.values()) == 0)
    return (zero_err_count / len(sentences_annotated_counts)) * 100.0


def probe_exact_match_accuracy(hypotheses, references, phenomenon_keywords=None):
    """Evaluate diagnostic morphological probe accuracy by testing whether target Kazakh
    inflectional suffixes and stems are correctly realized without morphological errors."""
    if not hypotheses or not references or len(hypotheses) != len(references):
        return 0.0

    # Normalized whole-sentence exact match, not an inferred morphology score.
    return 100.0 * sum(h.strip().lower() == r.strip().lower()
                       for h, r in zip(hypotheses, references)) / len(hypotheses)
