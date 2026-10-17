"""Clinical abbreviation lexicon.

Two sources, both independent of the evaluation data:
1. Mined from the supplied SNOMED dictionary: descriptions of the form
   "ABBR - Expansion" (e.g. "AR - Aortic regurgitation"), ~7.7k entries.
2. A small curated list of common chart shorthand that the dictionary does not
   spell out (HTN, CAD, LE, bilat, s/p, ...). Ambiguous abbreviations keep all
   readings; the ranker disambiguates with context.
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

from .dictionaries import load_snomed

CURATED = {
    "htn": ["hypertension"], "hld": ["hyperlipidemia"], "hl": ["hyperlipidemia", "hearing loss"],
    "dm": ["diabetes mellitus"], "dm2": ["type 2 diabetes mellitus"], "t2dm": ["type 2 diabetes mellitus"],
    "dmii": ["type 2 diabetes mellitus"], "niddm": ["type 2 diabetes mellitus"],
    "t1dm": ["type 1 diabetes mellitus"], "iddm": ["type 1 diabetes mellitus"],
    "cad": ["coronary artery disease"], "chf": ["congestive heart failure"], "hf": ["heart failure"],
    "hfref": ["heart failure with reduced ejection fraction"], "hfpef": ["heart failure with preserved ejection fraction"],
    "copd": ["chronic obstructive pulmonary disease"], "ckd": ["chronic kidney disease"],
    "esrd": ["end stage renal disease"], "aki": ["acute kidney injury"], "arf": ["acute renal failure"],
    "afib": ["atrial fibrillation"], "af": ["atrial fibrillation"], "a-fib": ["atrial fibrillation"],
    "aflutter": ["atrial flutter"], "svt": ["supraventricular tachycardia"], "vt": ["ventricular tachycardia"],
    "vtach": ["ventricular tachycardia"], "vfib": ["ventricular fibrillation"], "nsvt": ["nonsustained ventricular tachycardia"],
    "pvc": ["premature ventricular contraction"], "pvcs": ["premature ventricular contractions"],
    "pac": ["premature atrial contraction"], "lbbb": ["left bundle branch block"], "rbbb": ["right bundle branch block"],
    "dvt": ["deep vein thrombosis"], "pe": ["pulmonary embolism"], "mi": ["myocardial infarction"],
    "ami": ["acute myocardial infarction"], "nstemi": ["non st elevation myocardial infarction"],
    "stemi": ["st elevation myocardial infarction"], "cva": ["cerebrovascular accident"], "tia": ["transient ischemic attack"],
    "pad": ["peripheral arterial disease"], "pvd": ["peripheral vascular disease"], "aaa": ["abdominal aortic aneurysm"],
    "lvh": ["left ventricular hypertrophy"], "lvef": ["left ventricular ejection fraction"],
    "mr": ["mitral regurgitation"], "tr": ["tricuspid regurgitation"], "ar": ["aortic regurgitation"],
    "ai": ["aortic insufficiency"], "as": ["aortic stenosis"], "ms": ["multiple sclerosis", "mitral stenosis"],
    "mvp": ["mitral valve prolapse"], "pfo": ["patent foramen ovale"], "asd": ["atrial septal defect"],
    "uti": ["urinary tract infection"], "uri": ["upper respiratory infection"], "gerd": ["gastroesophageal reflux disease"],
    "gi": ["gastrointestinal"], "gib": ["gastrointestinal bleeding"], "ibs": ["irritable bowel syndrome"],
    "ibd": ["inflammatory bowel disease"], "osa": ["obstructive sleep apnea"], "bph": ["benign prostatic hyperplasia"],
    "ra": ["rheumatoid arthritis"], "oa": ["osteoarthritis"], "djd": ["degenerative joint disease"],
    "ddd": ["degenerative disc disease"], "sle": ["systemic lupus erythematosus"], "pid": ["pelvic inflammatory disease"],
    "hiv": ["human immunodeficiency virus"], "hcv": ["hepatitis c"], "hbv": ["hepatitis b"],
    "tb": ["tuberculosis"], "mrsa": ["methicillin resistant staphylococcus aureus"], "cdiff": ["clostridioides difficile"],
    "c diff": ["clostridioides difficile"], "ptsd": ["post traumatic stress disorder"], "adhd": ["attention deficit hyperactivity disorder"],
    "mdd": ["major depressive disorder"], "gad": ["generalized anxiety disorder"], "ocd": ["obsessive compulsive disorder"],
    "bpd": ["bipolar disorder", "borderline personality disorder"], "etoh": ["alcohol"], "sob": ["shortness of breath"],
    "doe": ["dyspnea on exertion"], "cp": ["chest pain"], "ha": ["headache"], "n/v": ["nausea and vomiting"],
    "nv": ["nausea and vomiting"], "abd": ["abdominal"], "loc": ["loss of consciousness"], "ams": ["altered mental status"],
    "le": ["lower extremity"], "lle": ["left lower extremity"], "rle": ["right lower extremity"],
    "ue": ["upper extremity"], "lue": ["left upper extremity"], "rue": ["right upper extremity"],
    "ble": ["bilateral lower extremity"], "bue": ["bilateral upper extremity"], "lbp": ["low back pain"],
    "bilat": ["bilateral"], "b/l": ["bilateral"], "bl": ["bilateral"], "lt": ["left"], "rt": ["right"],
    "l": ["left"], "r": ["right"], "hx": ["history"], "h/o": ["history of"], "fhx": ["family history"],
    "s/p": ["status post"], "w/": ["with"], "w/o": ["without"], "r/o": ["rule out"], "c/o": ["complains of"],
    "sx": ["symptoms"], "dx": ["diagnosis"], "fx": ["fracture"], "tx": ["treatment"], "pt": ["patient"],
    "ckd3": ["chronic kidney disease stage 3"], "ckd4": ["chronic kidney disease stage 4"],
    "ckd5": ["chronic kidney disease stage 5"], "hgb": ["hemoglobin"], "wbc": ["white blood cell"],
    "plt": ["platelet"], "bmi": ["body mass index"], "bp": ["blood pressure"], "hr": ["heart rate"],
    "rr": ["respiratory rate"], "ekg": ["electrocardiogram"], "ecg": ["electrocardiogram"], "echo": ["echocardiogram"],
    "cxr": ["chest x-ray"], "ct": ["computed tomography"], "mri": ["magnetic resonance imaging"],
    "ivc": ["inferior vena cava"], "svc": ["superior vena cava"], "lv": ["left ventricle"], "rv": ["right ventricle"],
    "la": ["left atrium"], "rvh": ["right ventricular hypertrophy"], "lad": ["left anterior descending artery"],
    "rca": ["right coronary artery"], "aov": ["aortic valve"], "mv": ["mitral valve"], "tv": ["tricuspid valve"],
    "pv": ["pulmonic valve"], "pulm": ["pulmonary"], "htn crisis": ["hypertensive crisis"],
    "ich": ["intracerebral hemorrhage"], "sah": ["subarachnoid hemorrhage"], "sdh": ["subdural hematoma"],
    "tbi": ["traumatic brain injury"], "ckd stage": ["chronic kidney disease stage"], "ugib": ["upper gastrointestinal bleeding"],
    "lgib": ["lower gastrointestinal bleeding"], "nash": ["nonalcoholic steatohepatitis"], "nafld": ["nonalcoholic fatty liver disease"],
    "ida": ["iron deficiency anemia"], "itp": ["immune thrombocytopenic purpura"], "dka": ["diabetic ketoacidosis"],
    "hhs": ["hyperosmolar hyperglycemic state"], "sirs": ["systemic inflammatory response syndrome"],
    "ards": ["acute respiratory distress syndrome"], "osteo": ["osteoarthritis", "osteomyelitis"],
    "dvt ppx": ["deep vein thrombosis prophylaxis"], "ppx": ["prophylaxis"], "rom": ["range of motion"],
    "adl": ["activities of daily living"], "adls": ["activities of daily living"], "iadl": ["instrumental activities of daily living"],
    "pt/ot": ["physical therapy occupational therapy"], "ot": ["occupational therapy"], "slp": ["speech language pathology"],
    "rw": ["rolling walker"], "fww": ["front wheeled walker"], "bsc": ["bedside commode"], "w/c": ["wheelchair"],
    "wc": ["wheelchair"], "sba": ["stand by assistance"], "cga": ["contact guard assistance"], "mod a": ["moderate assistance"],
    "min a": ["minimal assistance"], "max a": ["maximal assistance"], "wbat": ["weight bearing as tolerated"],
    "nwb": ["non weight bearing"], "tka": ["total knee arthroplasty"], "tkr": ["total knee replacement"],
    "tha": ["total hip arthroplasty"], "thr": ["total hip replacement"], "cabg": ["coronary artery bypass graft"],
    "pci": ["percutaneous coronary intervention"], "ptca": ["percutaneous transluminal coronary angioplasty"],
    "turp": ["transurethral resection of prostate"], "lap chole": ["laparoscopic cholecystectomy"],
    "egd": ["esophagogastroduodenoscopy"], "ercp": ["endoscopic retrograde cholangiopancreatography"],
    "hpi": ["history of present illness"], "ros": ["review of systems"], "nkda": ["no known drug allergies"],
    "pmh": ["past medical history"], "psh": ["past surgical history"], "prn": ["as needed"],
    "sig": ["significant"], "neg": ["negative"], "pos": ["positive"], "nl": ["normal"], "wnl": ["within normal limits"],
    "abn": ["abnormal"], "incr": ["increased"], "decr": ["decreased"], "dec": ["decreased"], "inc": ["increased"],
    "mod": ["moderate"], "sev": ["severe"], "chr": ["chronic"], "ac": ["acute", "acromioclavicular"],
    "lbs": ["pounds"], "yo": ["year old"], "y/o": ["year old"], "pna": ["pneumonia"], "cap": ["community acquired pneumonia"],
    "hcap": ["healthcare associated pneumonia"], "ssi": ["surgical site infection"], "cellulitis": ["cellulitis"],
    "dfu": ["diabetic foot ulcer"], "pu": ["pressure ulcer"], "ckd-mbd": ["chronic kidney disease mineral and bone disorder"],
    "hypoK": ["hypokalemia"], "hypok": ["hypokalemia"], "hypona": ["hyponatremia"], "hyperk": ["hyperkalemia"],
    "lft": ["liver function test"], "lfts": ["liver function tests"], "bun": ["blood urea nitrogen"], "cr": ["creatinine"],
    "gfr": ["glomerular filtration rate"], "egfr": ["estimated glomerular filtration rate"], "a1c": ["hemoglobin a1c"],
    "hba1c": ["hemoglobin a1c"], "tsh": ["thyroid stimulating hormone"], "psa": ["prostate specific antigen"],
    "inr": ["international normalized ratio"], "ptt": ["partial thromboplastin time"], "cbc": ["complete blood count"],
    "bmp": ["basic metabolic panel"], "cmp": ["comprehensive metabolic panel"], "ua": ["urinalysis"],
    "rbc": ["red blood cell"], "mcv": ["mean corpuscular volume"], "esr": ["erythrocyte sedimentation rate"],
    "crp": ["c reactive protein"], "bnp": ["brain natriuretic peptide"], "trop": ["troponin"],
    "hep c": ["hepatitis c"], "hep b": ["hepatitis b"], "ca": ["cancer", "calcium"], "mets": ["metastases"],
    "nsclc": ["non small cell lung cancer"], "sclc": ["small cell lung cancer"], "crc": ["colorectal cancer"],
    "hcc": ["hepatocellular carcinoma"], "cll": ["chronic lymphocytic leukemia"], "aml": ["acute myeloid leukemia"],
    "mgus": ["monoclonal gammopathy of undetermined significance"], "rcc": ["renal cell carcinoma"],
    "bcc": ["basal cell carcinoma"], "scc": ["squamous cell carcinoma"], "dcis": ["ductal carcinoma in situ"],
    "ptx": ["pneumothorax"], "rll": ["right lower lobe"], "rul": ["right upper lobe"], "rml": ["right middle lobe"],
    "lll": ["left lower lobe"], "lul": ["left upper lobe"], "ggo": ["ground glass opacity"],
    "sbo": ["small bowel obstruction"], "luq": ["left upper quadrant"], "ruq": ["right upper quadrant"],
    "llq": ["left lower quadrant"], "rlq": ["right lower quadrant"], "rom limited": ["limited range of motion"],
    "tmj": ["temporomandibular joint"], "cts": ["carpal tunnel syndrome"], "plantar fasc": ["plantar fasciitis"],
    "rls": ["restless legs syndrome"], "pd": ["parkinson disease"], "als": ["amyotrophic lateral sclerosis"],
    "mci": ["mild cognitive impairment"], "ad": ["alzheimer disease"], "sz": ["seizure"], "szs": ["seizures"],
    "gtc": ["generalized tonic clonic"], "dm neuropathy": ["diabetic neuropathy"], "pn": ["peripheral neuropathy"],
    "esld": ["end stage liver disease"], "sbp": ["spontaneous bacterial peritonitis", "systolic blood pressure"],
    "dbp": ["diastolic blood pressure"], "hfr": ["heart failure"], "cm": ["cardiomyopathy"],
    "icm": ["ischemic cardiomyopathy"], "nicm": ["nonischemic cardiomyopathy"], "hocm": ["hypertrophic obstructive cardiomyopathy"],
    "ppm": ["permanent pacemaker"], "aicd": ["automatic implantable cardioverter defibrillator"], "icd": ["implantable cardioverter defibrillator"],
    "pih": ["pregnancy induced hypertension"], "gdm": ["gestational diabetes mellitus"], "pcos": ["polycystic ovary syndrome"],
    "aub": ["abnormal uterine bleeding"], "std": ["sexually transmitted disease"], "sti": ["sexually transmitted infection"],
    "ed": ["erectile dysfunction"], "oab": ["overactive bladder"], "lut": ["lower urinary tract"], "luts": ["lower urinary tract symptoms"],
    "ckd 3": ["chronic kidney disease stage 3"], "stg": ["stage"], "stag": ["stage"], "hep": ["hepatitis"],
    "vit d": ["vitamin d"], "vit b12": ["vitamin b12"], "b12": ["vitamin b12"], "obs": ["obesity"],
    "morbid obesity": ["morbid obesity"], "ftt": ["failure to thrive"], "fsgs": ["focal segmental glomerulosclerosis"],
}

_ABBR_DESC = re.compile(r"^([A-Za-z][A-Za-z0-9/\-]{0,7}) - (.+)$")


@lru_cache(maxsize=1)
def lexicon() -> dict[str, list[str]]:
    lex: dict[str, list[str]] = defaultdict(list)
    for k, v in CURATED.items():
        lex[k.lower()].extend(v)
    sn = load_snomed()
    for d in sn["description"]:
        m = _ABBR_DESC.match(d)
        if not m:
            continue
        ab, exp = m.group(1).lower(), m.group(2).lower().strip()
        if len(ab) < 2 or exp in lex[ab] or len(lex[ab]) >= 6:
            continue
        lex[ab].append(exp)
    return dict(lex)


_TOK = re.compile(r"[a-z0-9]+(?:/[a-z0-9]+)?|[^\sa-z0-9]")


def expand(text: str, max_variants: int = 3) -> list[str]:
    """Return up to max_variants expanded forms of a normalised (lowercase) string.

    Only tokens present in the lexicon are rewritten; the first reading of each
    abbreviation forms variant 1, alternative readings of the first ambiguous
    abbreviation form further variants.
    """
    lex = lexicon()
    toks = text.split()
    if not toks:
        return []
    out_main, alt_slot = [], None
    changed = False
    for i, t in enumerate(toks):
        core = t.strip(",.;:()[]")
        if core in lex and not core.isdigit():
            out_main.append(lex[core][0])
            changed = True
            if alt_slot is None and len(lex[core]) > 1:
                alt_slot = (i, lex[core][1:])
        else:
            out_main.append(t)
    if not changed:
        return []
    variants = [" ".join(out_main)]
    if alt_slot:
        i, alts = alt_slot
        for a in alts[: max_variants - 1]:
            v = list(out_main)
            v[i] = a
            variants.append(" ".join(v))
    return variants[:max_variants]
