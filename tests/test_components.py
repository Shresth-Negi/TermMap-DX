"""Fast unit tests for the deterministic building blocks (no models, no GPU).

Run: source scripts/env.sh && $PY -m pytest -q tests
Tests that need the challenge dictionaries are skipped when data/raw is absent.
"""
import pandas as pd
import pytest

from termmap.features import clean_entity, context_flags, context_without_mention, lat_of, numbers
from termmap.io import load_input
from termmap.normalize import basic_norm, key_norm, strip_semtag
from termmap.paths import RAW

HAVE_DATA = (RAW / "icd10cm_codes.xlsx").exists()


def test_basic_norm_strips_markup_and_case():
    assert basic_norm("Diabetes<br>Mellitus  Type 2") == "diabetes mellitus type 2"
    assert basic_norm("a &amp; b") == "a & b"
    assert key_norm("CKD, stage-3!") == "ckd stage 3"


def test_strip_semtag():
    assert strip_semtag("Chest pain (finding)") == "chest pain"
    assert strip_semtag("Stress incontinence (female) (male)") == "stress incontinence (female)"


def test_clean_entity_removes_ehr_artifacts_and_finds_codes():
    text, codes = clean_entity("Primary osteoarthritis of left knee M17.12 (HCC)")
    assert "(hcc)" not in text
    assert codes == ["M17.12"]
    assert clean_entity("- 5) aov sclerosis")[0] == "aov sclerosis"


def test_laterality():
    assert lat_of("left knee oa") == 1
    assert lat_of("rt shoulder pain") == 2
    assert lat_of("bilateral lower extremity edema") == 3
    assert lat_of("left and right") == 3
    assert lat_of("knee pain") == 0


def test_numbers_include_roman_numerals():
    assert numbers("stage 2 mild ckd") == {"2"}
    assert numbers("type ii diabetes") == {"2"}
    assert numbers("CKD stage 03") == {"3"}


def test_context_flags():
    assert context_flags("chest pain", "ROS: denies chest pain or palpitations")[0] == 1
    assert context_flags("h/o dvt", "pmh: h/o dvt 2015")[1] == 1
    assert context_flags("breast cancer", "family history: mother with breast cancer")[2] == 1
    assert context_flags("cough", "productive cough for 3 days") == (0, 0, 0)


def test_context_without_mention():
    w = context_without_mention("MS", "echo: severe MS with mitral valve area 0.9")
    assert "ms" not in w.split() and "mitral" in w


def test_load_input_accepts_brief_schema(tmp_path):
    p = tmp_path / "in.csv"
    pd.DataFrame({"chunk_id": ["c1"], "clinical_entity": ["HTN"]}).to_csv(p, index=False)
    df = load_input(str(p))
    assert list(df["entity"]) == ["HTN"]
    assert "chunk_text" in df and "subconcept" in df


@pytest.mark.skipif(not HAVE_DATA, reason="challenge dictionaries not present")
def test_abbreviation_lexicon_keeps_all_readings():
    from termmap.abbrev import expand, lexicon
    lex = lexicon()
    assert "hypertension" in lex["htn"]
    assert {"multiple sclerosis", "mitral stenosis"} <= set(lex["ms"])
    assert expand("htn") == ["hypertension"]
    assert len(expand("ms")) >= 2


@pytest.mark.skipif(not HAVE_DATA, reason="challenge dictionaries not present")
def test_icd_official_titles_and_billable():
    from termmap.corpus import description_map
    from termmap.dictionaries import load_icd
    d = description_map("ICD10CM")
    assert d["R06.89"] == "Other abnormalities of breathing"
    assert d["I10"] == "Essential (primary) hypertension"
    icd = load_icd().drop_duplicates("code").set_index("code")["billable"]
    assert bool(icd["E11.9"]) and not bool(icd["E11"])
