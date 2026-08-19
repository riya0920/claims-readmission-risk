"""Charlson Comorbidity Index, Quan (2005) ICD-10 adaptation.

Two things a payer screener checks here:

1. That the index is computed from a DIAGNOSIS HISTORY WINDOW, not from the
   index admission's own diagnoses. Comorbidity is what the member carried in
   with them. Scoring the index stay's dx list means the acute event inflates
   the chronic burden score, and (worse) codes assigned during the stay are
   only fully known once the stay is adjudicated -- which is after the point
   we score at. This module takes a pre-filtered code list and does not know
   about dates, so the caller carries that responsibility; features.py does.

2. The hierarchy rules. Quan's definition is not a plain sum of flags:
   - diabetes with complication supersedes diabetes without (score 2, not 1+2)
   - severe liver disease supersedes mild (score 3, not 1+3)
   - metastatic solid tumour supersedes any/malignancy (score 6, not 2+6)
   Getting the hierarchy wrong is the most common quiet error in Charlson
   implementations, and it inflates the score for exactly the sickest members.

Weights are the original Charlson weights as carried forward by Quan. The
Quan-updated weight set (which collapses several conditions to 0) is a
defensible alternative; this implementation uses the classic weights and says
so, because "which weight set" is a real question and picking silently is the
failure.
"""

# condition -> (weight, [ICD-10 code prefixes])
# Prefixes follow Quan et al. 2005, Med Care 43(11):1130-9, abbreviated to the
# code families this dataset emits. A production implementation loads the full
# Quan prefix list (~300 prefixes); this covers ~60 and is honest about it.
CHARLSON_DEFINITION = {
    "myocardial_infarction": (1, ["I21", "I22", "I252"]),
    "congestive_heart_failure": (1, [
        "I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426",
        "I427", "I428", "I429", "I43", "I50", "P290",
    ]),
    "peripheral_vascular": (1, [
        "I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792",
        "K551", "K558", "K559", "Z958", "Z959",
    ]),
    "cerebrovascular": (1, ["G45", "G46", "H340", "I60", "I61", "I62", "I63",
                            "I64", "I65", "I66", "I67", "I68", "I69"]),
    "dementia": (1, ["F00", "F01", "F02", "F03", "F051", "G30", "G311"]),
    "chronic_pulmonary": (1, [
        "I278", "I279", "J40", "J41", "J42", "J43", "J44", "J45", "J46",
        "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67",
        "J684", "J701", "J703",
    ]),
    "rheumatic": (1, ["M05", "M06", "M315", "M32", "M33", "M34", "M351",
                      "M353", "M360"]),
    "peptic_ulcer": (1, ["K25", "K26", "K27", "K28"]),
    "mild_liver": (1, [
        "B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714",
        "K715", "K717", "K73", "K74", "K760", "K762", "K763", "K764",
        "K768", "K769", "Z944",
    ]),
    "diabetes_uncomplicated": (1, ["E100", "E101", "E106", "E108", "E109",
                                   "E110", "E111", "E116", "E118", "E119",
                                   "E120", "E130", "E140"]),
    "diabetes_complicated": (2, ["E102", "E103", "E104", "E105", "E107",
                                 "E112", "E113", "E114", "E115", "E117",
                                 "E132", "E133", "E134", "E135", "E142"]),
    "hemiplegia": (2, ["G041", "G114", "G801", "G802", "G81", "G82", "G830",
                       "G831", "G832", "G833", "G834", "G839"]),
    "renal": (2, ["I120", "I131", "N032", "N033", "N034", "N035", "N036",
                  "N037", "N052", "N053", "N054", "N055", "N056", "N057",
                  "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940",
                  "Z992"]),
    "malignancy": (2, [
        "C0", "C1", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
        "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40",
        "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51",
        "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61",
        "C62", "C63", "C64", "C65", "C66", "C67", "C68", "C69", "C70",
        "C71", "C72", "C73", "C74", "C75", "C76", "C81", "C82", "C83",
        "C84", "C85", "C88", "C90", "C91", "C92", "C93", "C94", "C95",
        "C96", "C97",
    ]),
    "severe_liver": (3, ["I850", "I859", "I864", "I982", "K704", "K711",
                         "K721", "K729", "K765", "K766", "K767"]),
    "metastatic_tumour": (6, ["C77", "C78", "C79", "C80"]),
    "aids_hiv": (6, ["B20", "B21", "B22", "B24"]),
}

# superseding condition -> condition whose weight it cancels
HIERARCHY = {
    "diabetes_complicated": "diabetes_uncomplicated",
    "severe_liver": "mild_liver",
    "metastatic_tumour": "malignancy",
}


def _normalise(code):
    """ICD-10-CM codes arrive dotted (E11.22); Quan prefixes are undotted."""
    return code.upper().replace(".", "").strip()


def charlson_flags(codes):
    """Return {condition: bool} for a list of ICD-10-CM codes, hierarchy applied."""
    norm = [_normalise(c) for c in codes]
    flags = {}
    for cond, (_weight, prefixes) in CHARLSON_DEFINITION.items():
        flags[cond] = any(n.startswith(p) for n in norm for p in prefixes)
    for superseding, superseded in HIERARCHY.items():
        if flags[superseding]:
            flags[superseded] = False
    return flags


def charlson_score(codes):
    """Charlson Comorbidity Index (classic weights, Quan ICD-10 code sets)."""
    flags = charlson_flags(codes)
    return sum(w for cond, (w, _p) in CHARLSON_DEFINITION.items() if flags[cond])


def charlson_detail(codes):
    """(score, {condition: weight}) for the conditions that fired -- used by the
    worklist to explain a score to a care manager in condition names, not
    a number."""
    flags = charlson_flags(codes)
    hits = {c: CHARLSON_DEFINITION[c][0] for c in flags if flags[c]}
    return sum(hits.values()), hits
