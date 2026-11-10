"""Provisional error labels for the 30 lowest-chrF2 EN->KK outputs (first pass; must be re-labelled blind by annotators).
Usage: python label_errors.py <assignment4-dir>"""
import csv, json, re, sys
from pathlib import Path
D = Path(sys.argv[1])
L = {
 "mt_flores200_00874": ("Script", "Whole output in Latin-script Kazakh (Qatti aua raiı…); reference is Cyrillic"),
 "mt_flores200_00088": ("Script", "Latin-script output; also invented name Saǵıshegish for Sagittarius"),
 "mt_flores200_00303": ("Script", "Mixed Latin/Cyrillic; 'Economy, Trade and Industry' left in English"),
 "mt_flores200_00446": ("Reference mismatch", "Correct meaning; nouns (пацифизм мен оқшаулану) instead of reference adjectives"),
 "mt_flores200_00860": ("Lexical", "Latin word 'Sol' for 'Осы'; non-existent word қағазбастылық"),
 "mt_flores200_00886": ("Reference mismatch", "Correct paraphrase (қатаңдай түсті)"),
 "mt_flores200_00346": ("Morphological", "Possessive suffix error: әрпи instead of әрпі"),
 "mt_flores200_00201": ("Lexical", "Super-G rendered as супер-гигант; name forms differ (Аустралиялық, Гурли)"),
 "mt_flores200_00480": ("Reference mismatch", "Output is faithful; reference itself is loose"),
 "mt_flores200_00865": ("Semantic", "Negation scope: 'not all Sámi' became 'all Sámi did not'"),
 "mt_flores200_00502": ("Semantic", "Added gloss not in source: (арнайы сөздік қорын)"),
 "mt_flores200_00839": ("Reference mismatch", "Acceptable restructuring of the sentence"),
 "mt_flores200_00666": ("Lexical", "Citation names left in Latin script; команда vs топ"),
 "mt_flores200_00725": ("Reference mismatch", "Correct paraphrase"),
 "mt_flores200_00202": ("Lexical", "Non-existent numeral беінші (should be бесінші)"),
 "mt_flores200_00210": ("Reference mismatch", "Correct, clause order changed"),
 "mt_flores200_00868": ("Semantic", "'Subject to a travel advisory' rendered as 'travel is recommended'"),
 "mt_flores200_00900": ("Reference mismatch", "Correct paraphrase"),
 "mt_flores200_00952": ("Semantic", "'Worked to death' rendered as 'tortured to death'"),
 "mt_flores200_00060": ("Lexical", "St. Peter's as Әуес Петр (wrong word for 'saint', should be Әулие)"),
 "mt_flores200_00972": ("Reference mismatch", "Correct paraphrase"),
 "mt_flores200_00776": ("Lexical", "Tamaki Drive left in Latin; misspelt Мишшн"),
 "mt_flores200_00867": ("Lexical", "Russian spelling Цыган instead of Kazakh Сыған"),
 "mt_flores200_00765": ("Lexical", "Hallucinated gloss (лосось/моос): лосось is Russian for 'salmon'"),
 "mt_flores200_00436": ("Reference mismatch", "Correct; reference splits into two sentences"),
 "mt_flores200_00856": ("Reference mismatch", "Correct overall; арқалық for 'backpack' is unusual"),
 "mt_flores200_00150": ("Reference mismatch", "Correct; translates the MSF name, reference keeps French"),
 "mt_flores200_00288": ("Reference mismatch", "Correct paraphrase"),
 "mt_flores200_00384": ("Lexical", "Garbled non-word сондай-жекөлікке ('also the water')"),
 "mt_flores200_00808": ("Reference mismatch", "Correct paraphrase"),
}
rows = list(csv.DictReader(open(D / "results/error_candidates_en2kk.csv", encoding="utf-8")))
assert len(rows) == 30 and all(r["item_id"] in L for r in rows)
for r in rows:
    r["error_type"], r["note"] = L[r["item_id"]]
with open(D / "error_annotation.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# script check over all EN->KK outputs
per = list(csv.DictReader(open(D / "results/per_item_scores.csv", encoding="utf-8")))
ek = [r for r in per if r["direction"] == "en2kk"]
def latin_share(s):
    lat = len(re.findall(r"[A-Za-zÀ-ɏ]", s)); cyr = len(re.findall(r"[Ѐ-ӿ]", s))
    return lat / (lat + cyr) if lat + cyr else 0
mostly_latin = [r["item_id"] for r in ek if latin_share(r["hyp"]) > 0.5]
json.dump({"en2kk_outputs": len(ek), "mostly_latin_script_outputs": mostly_latin, "count": len(mostly_latin)},
          open(D / "results/script_check.json", "w"), indent=2)
print(len(mostly_latin), mostly_latin)
