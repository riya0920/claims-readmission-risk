"""ICD-10-CM / CPT / NDC code tables and a CCSR-style grouper.

HONEST SCOPE NOTE
-----------------
The official CCSR (Clinical Classifications Software Refined) grouper is an
HCUP-published CSV mapping all ~70,000 ICD-10-CM codes to ~530 categories.
It is not vendored here (offline build). What follows is a hand-curated
subset that follows the CCSR *convention* -- a body-system prefix plus a
three-digit ordinal (CIR###, END###, RSP###, ...) -- covering the conditions
this generator emits. Category codes here are illustrative, NOT authoritative
HCUP category IDs. In production you load the HCUP file and version-pin it,
because CCSR is re-released annually and category membership moves.

The point being demonstrated is the modelling decision (group before you
model, because raw ICD-10-CM is too sparse to learn from), not the contents
of a file that would be a download in any real environment.
"""

# ---------------------------------------------------------------------------
# ICD-10-CM codes this simulation emits, grouped by clinical condition.
# Real ICD-10-CM codes, chosen because they are the ones that actually drive
# readmission risk in a payer population.
# ---------------------------------------------------------------------------
CONDITION_CODES = {
    "chf":        ["I50.22", "I50.32", "I50.42", "I50.9", "I11.0"],
    "copd":       ["J44.0", "J44.1", "J44.9", "J43.9"],
    "diabetes":   ["E11.9", "E11.65", "E10.9"],
    "diabetes_c": ["E11.22", "E11.42", "E11.51", "E11.65", "E10.22"],
    "ckd":        ["N18.3", "N18.4", "N18.5", "N18.6"],
    "cad_mi":     ["I25.10", "I21.4", "I21.9", "I25.2"],
    "afib":       ["I48.0", "I48.91"],
    "cancer":     ["C34.90", "C50.911", "C18.9", "C61"],
    "mets":       ["C78.00", "C79.51", "C77.9"],
    "cva":        ["I63.9", "I69.30", "G45.9"],
    "dementia":   ["F03.90", "G30.9"],
    "liver_mild": ["K70.30", "K74.60", "B18.2"],
    "liver_sev":  ["K72.90", "I85.00", "K76.7"],
    "pvd":        ["I73.9", "I70.219"],
    "pud":        ["K25.9", "K27.9"],
    "rheum":      ["M06.9", "M32.9", "M05.79"],
    "hemiplegia": ["G81.90", "G82.20"],
    "hiv":        ["B20"],
    "psych":      ["F32.9", "F20.9", "F41.9"],
    "sud":        ["F11.20", "F10.20"],
    # acute / presenting problems
    "pneumonia":  ["J18.9", "J15.9"],
    "sepsis":     ["A41.9", "R65.20"],
    "aki":        ["N17.9"],
    "uti":        ["N39.0"],
    "cellulitis": ["L03.115"],
    "fall":       ["W19.XXXA", "S72.001A"],
    "chest_pain": ["R07.9"],
    "syncope":    ["R55"],
}

# ---------------------------------------------------------------------------
# CCSR-style grouper: ICD-10-CM -> (category code, category description).
# Keyed by code prefix, longest-prefix-wins.
# ---------------------------------------------------------------------------
_CCSR_PREFIX = [
    ("I50",   "CIR019", "Heart failure"),
    ("I11.0", "CIR019", "Heart failure"),
    ("I21",   "CIR009", "Acute myocardial infarction"),
    ("I25",   "CIR011", "Coronary atherosclerosis and other heart disease"),
    ("I48",   "CIR017", "Cardiac dysrhythmias"),
    ("I63",   "CIR020", "Acute cerebrovascular disease"),
    ("I69",   "CIR021", "Late effects of cerebrovascular disease"),
    ("G45",   "CIR020", "Acute cerebrovascular disease"),
    ("I70",   "CIR024", "Peripheral and visceral vascular disease"),
    ("I73",   "CIR024", "Peripheral and visceral vascular disease"),
    ("I85",   "DIG019", "Liver disease: other (portal hypertension)"),
    ("J44",   "RSP008", "Chronic obstructive pulmonary disease"),
    ("J43",   "RSP008", "Chronic obstructive pulmonary disease"),
    ("J18",   "RSP002", "Pneumonia (except that caused by TB)"),
    ("J15",   "RSP002", "Pneumonia (except that caused by TB)"),
    ("E11.2", "END005", "Diabetes mellitus with complication"),
    ("E11.4", "END005", "Diabetes mellitus with complication"),
    ("E11.5", "END005", "Diabetes mellitus with complication"),
    ("E11.6", "END005", "Diabetes mellitus with complication"),
    ("E10.2", "END005", "Diabetes mellitus with complication"),
    ("E11",   "END004", "Diabetes mellitus without complication"),
    ("E10",   "END004", "Diabetes mellitus without complication"),
    ("N18",   "GEN003", "Chronic kidney disease"),
    ("N17",   "GEN002", "Acute kidney injury"),
    ("N39.0", "GEN004", "Urinary tract infections"),
    ("C34",   "NEO015", "Respiratory cancers"),
    ("C50",   "NEO018", "Breast cancer"),
    ("C18",   "NEO011", "Colorectal cancer"),
    ("C61",   "NEO029", "Prostate cancer"),
    ("C77",   "NEO070", "Secondary malignancies"),
    ("C78",   "NEO070", "Secondary malignancies"),
    ("C79",   "NEO070", "Secondary malignancies"),
    ("F03",   "MBD025", "Neurocognitive disorders"),
    ("G30",   "MBD025", "Neurocognitive disorders"),
    ("F32",   "MBD002", "Depressive disorders"),
    ("F20",   "MBD006", "Schizophrenia spectrum disorders"),
    ("F41",   "MBD005", "Anxiety and fear-related disorders"),
    ("F11",   "MBD018", "Opioid-related disorders"),
    ("F10",   "MBD017", "Alcohol-related disorders"),
    ("K70",   "DIG018", "Liver disease: alcohol-associated"),
    ("K74",   "DIG019", "Liver disease: other"),
    ("B18",   "INF008", "Viral hepatitis"),
    ("K72",   "DIG020", "Hepatic failure"),
    ("K76",   "DIG019", "Liver disease: other"),
    ("K25",   "DIG010", "Peptic ulcer disease"),
    ("K27",   "DIG010", "Peptic ulcer disease"),
    ("M06",   "MUS006", "Rheumatoid arthritis and related disease"),
    ("M05",   "MUS006", "Rheumatoid arthritis and related disease"),
    ("M32",   "MUS010", "Systemic lupus erythematosus"),
    ("G81",   "NVS020", "Paralysis (hemiplegia/hemiparesis)"),
    ("G82",   "NVS020", "Paralysis (paraplegia/quadriplegia)"),
    ("B20",   "INF006", "HIV infection"),
    ("A41",   "INF003", "Sepsis"),
    ("R65",   "INF003", "Sepsis"),
    ("L03",   "SKN001", "Skin and subcutaneous tissue infections"),
    ("W19",   "INJ031", "Falls"),
    ("S72",   "INJ008", "Fracture of the lower limb"),
    ("R07",   "SYM006", "Chest pain"),
    ("R55",   "SYM008", "Syncope"),
]
# longest prefix first so E11.2 beats E11
_CCSR_PREFIX.sort(key=lambda r: -len(r[0]))

CCSR_UNMAPPED = ("XXX000", "Unmapped")


def ccsr(code):
    """Map an ICD-10-CM code to (category, description). Longest prefix wins."""
    c = code.upper().strip()
    for prefix, cat, desc in _CCSR_PREFIX:
        if c.startswith(prefix):
            return cat, desc
    return CCSR_UNMAPPED


def ccsr_category(code):
    return ccsr(code)[0]


# ---------------------------------------------------------------------------
# CPT/HCPCS procedure codes, by setting.
# ---------------------------------------------------------------------------
CPT = {
    "ed_visit":     ["99283", "99284", "99285"],
    "office_visit": ["99213", "99214", "99215"],
    "ip_initial":   ["99221", "99222", "99223"],
    "ip_subseq":    ["99231", "99232", "99233"],
    "ip_discharge": ["99238", "99239"],
    "dialysis":     ["90960", "90961"],
    "chemo":        ["96413", "96415"],
    "lab_bmp":      ["80048"],
    "lab_a1c":      ["83036"],
    "echo":         ["93306"],
    "xray_chest":   ["71046"],
    "home_health":  ["G0299", "G0300"],
    "tcm_visit":    ["99495", "99496"],
}

# CMS place-of-service codes
POS = {
    "office": "11",
    "home": "12",
    "inpatient_hospital": "21",
    "outpatient_hospital": "22",
    "emergency_room": "23",
    "asc": "24",
    "snf": "31",
    "telehealth": "02",
}

# Discharge disposition (UB-04 patient status). NOTE: this field is the
# planted leakage vector -- see docs/LEAKAGE_AUDIT.md. It is populated at
# claim adjudication, i.e. it encodes what happened AFTER the discharge
# decision point we score at.
DISCHARGE_STATUS = {
    "01": "Home / self care",
    "03": "Skilled nursing facility",
    "06": "Home health service",
    "07": "Left against medical advice",
    "20": "Expired",
    "62": "Inpatient rehabilitation facility",
}

# Therapeutic classes -> representative NDCs (11-digit, format-valid;
# the real NDC directory is a download, these are shape-valid stand-ins).
NDC = {
    "loop_diuretic":  ["00093-0742-01", "00378-0208-05"],
    "beta_blocker":   ["00093-0733-01", "00378-0018-10"],
    "acei_arb":       ["00093-1029-01", "00378-0264-05"],
    "insulin":        ["00002-8215-01", "00169-1834-11"],
    "metformin":      ["00093-1048-01", "00378-0242-05"],
    "statin":         ["00093-7152-01", "00378-3925-05"],
    "anticoagulant":  ["00093-0106-01", "00378-5175-05"],
    "inhaled_copd":   ["00173-0682-20", "00093-0311-01"],
    "opioid":         ["00093-0032-01"],
    "antipsychotic":  ["00093-0116-01"],
}

CHRONIC_CLASSES = {
    "chf": ["loop_diuretic", "beta_blocker", "acei_arb"],
    "copd": ["inhaled_copd"],
    "diabetes": ["metformin"],
    "diabetes_c": ["insulin", "metformin"],
    "cad_mi": ["statin", "beta_blocker"],
    "afib": ["anticoagulant"],
    "ckd": ["acei_arb"],
    "psych": ["antipsychotic"],
    "sud": ["opioid"],
}
