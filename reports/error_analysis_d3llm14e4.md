# Error analysis (out-of-fold, tag `_d3llm14e4`)


## ICD10CM  (n=2306)

* exact match: **0.7082**  (stage-1 only: 0.6531)
* same 3-char category: 0.8122, same chapter: 0.8929

| gold-code frequency elsewhere in train | accuracy | n |
|---|---|---|
| unseen (0) | 0.4571 | 466 |
| rare (1-2) | 0.7041 | 517 |
| mid (3-9) | 0.7820 | 899 |
| frequent (10+) | 0.8325 | 424 |

| failure mode | rows | share |
|---|---|---|
| correct | 1633 | 0.7082 |
| retrieval_miss | 200 | 0.0867 |
| specificity (same 3-char category) | 193 | 0.0837 |
| wrong concept, other chapter | 144 | 0.0624 |
| wrong concept, same chapter | 111 | 0.0481 |
| laterality | 25 | 0.0108 |

| coverage (most confident) | accuracy |
|---|---|
| 100% | 0.7082 |
| 97% | 0.7211 |
| 90% | 0.7494 |
| 80% | 0.7881 |
| 70% | 0.8259 |
| 50% | 0.8812 |

| subconcept | accuracy | n |
|---|---|---|
| DiagnosisCode | 0.7085 | 2305 |
| ProcedureCode | 0.0000 | 1 |

Sample errors:


**laterality**

- `impacted cerumen` → gold `H61.21` Impacted cerumen, right ear | pred `H61.20` Impacted cerumen, unspecified ear
- `circumferential cellulitis of the lower leg` → gold `L03.116` Cellulitis of left lower limb | pred `L03.119` Cellulitis of unspecified part of limb
- `rv dysfunction` → gold `I51.9` Heart disease, unspecified | pred `I50.810` Right heart failure without mention of left heart failure
- `premedication of other armsequo osteoarthritis` → gold `M17.0` Bilateral primary osteoarthritis of knee | pred `M17.9` Osteoarthritis of knee, unspecified
- `mild ica disease b/l` → gold `I65.23` Occlusion and stenosis of bilateral carotid arteries | pred `I65.29` Occlusion and stenosis of unspecified carotid artery
- `swelling of the legs` → gold `R22.43` Localized swelling, mass and lump, lower limb, bilateral | pred `R22.40` Localized swelling, mass and lump, unspecified lower limb

**retrieval_miss**

- `bacterial endocarditis` → gold `A49.01` Methicillin susceptible Staphylococcus aureus infection, unspecified site | pred `I33.0` Purulent endocarditis (acute) (subacute)
- `stabbing` → gold `R52` Pain, unspecified | pred `R25.2` Cramp and spasm
- `lvad disconnection` → gold `T82.598A` Other mechanical complication of other cardiac and vascular devices and implants, initial encounter | pred `Z95.811` Presence of heart assist device
- `lad` → gold `R59.0` Localized enlarged lymph nodes | pred `I25.10` Atherosclerotic heart disease of native coronary artery without angina pectoris
- `temporary paralysis` → gold `G97.82` Other postprocedural complications and disorders of nervous system | pred `R29.5` Transient paralysis
- `tissue injury due to loss of protective sensation` → gold `L98.9` Disorder of the skin and subcutaneous tissue, unspecified | pred `R20.8` Other disturbances of skin sensation

**specificity (same 3-char category)**

- `mild to moderate bph` → gold `N40.1` Enlarged prostate with LUTS | pred `N40.0` Benign prostatic hyperplasia without lower urinary tract symptoms
- `open wound of left knee, leg, and ankle, initial encounter` → gold `S81.802A` Unspecified open wound, left lower leg, initial encounter | pred `S81.002A` Unspecified open wound, left knee, initial encounter
- `decreased smell` → gold `R43.8` Other disturbances of smell and taste | pred `R43.9` Unspecified disturbances of smell and taste
- `nasal dyspnea` → gold `R06.89` Other abnormalities of breathing | pred `R06.09` Other forms of dyspnea
- `sensory loss` → gold `R20.0` Anesthesia of skin | pred `R20.9` Unspecified disturbances of skin sensation
- `more difficulty with stairs` → gold `R26.89` Other abnormalities of gait and mobility | pred `R26.2` Difficulty in walking, not elsewhere classified

**wrong concept, other chapter**

- `aos` → gold `I35.0` Nonrheumatic aortic (valve) stenosis | pred `Z03.89` Encounter for observation for other suspected diseases and conditions ruled out
- `neurogenic claudication` → gold `M48.062` Spinal stenosis, lumbar region with neurogenic claudication | pred `I73.9` Peripheral vascular disease, unspecified
- `post fall` → gold `Z91.81` History of falling | pred `W19.XXXA` Unspecified fall, initial encounter
- `did not sleep last night` → gold `G47.00` Insomnia, unspecified | pred `Z72.820` Lack of adequate sleep
- `resolving hhs` → gold `E11.00` Type 2 diabetes mellitus with hyperosmolarity without nonketotic hyperglycemic-hyperosmolar coma (NKHHC) | pred `R29.898` Other symptoms and signs involving the musculoskeletal system
- `suspected infection` → gold `Z03.89` Encounter for observation for other suspected diseases and conditions ruled out | pred `B99.9` Unspecified infectious disease

**wrong concept, same chapter**

- `significant vascular disease` → gold `I99.8` Other disorder of circulatory system | pred `I73.9` Peripheral vascular disease, unspecified
- `bacterial pneumonia` → gold `J15.9` Unspecified bacterial pneumonia | pred `J18.9` Pneumonia, unspecified organism
- `multilevel degenerative changes` → gold `M50.30` Other cervical disc degeneration, unspecified cervical region | pred `M47.819` Spondylosis without myelopathy or radiculopathy, site unspecified
- `abnormal movement` → gold `R29.898` Other symptoms and signs involving the musculoskeletal system | pred `R25.9` Unspecified abnormal involuntary movements
- `degenerative joint disease (djd) of lumbar spine` → gold `M47.816` Spondylosis without myelopathy or radiculopathy, lumbar region | pred `M51.36` Other intervertebral disc degeneration, lumbar region
- `stiff neck` → gold `M43.6` Torticollis | pred `M25.60` Stiffness of unspecified joint, not elsewhere classified

## SNOMEDCT_US  (n=7340)

* exact match: **0.5816**  (stage-1 only: 0.5153)

| gold-code frequency elsewhere in train | accuracy | n |
|---|---|---|
| unseen (0) | 0.4433 | 2964 |
| rare (1-2) | 0.6140 | 2308 |
| mid (3-9) | 0.7392 | 1764 |
| frequent (10+) | 0.7697 | 304 |

| failure mode | rows | share |
|---|---|---|
| correct | 4269 | 0.5816 |
| retrieval_miss | 1094 | 0.1490 |
| semantic-type mismatch (e.g. finding vs procedure) | 975 | 0.1328 |
| wrong concept, same semantic type | 943 | 0.1285 |
| laterality | 59 | 0.0080 |

| coverage (most confident) | accuracy |
|---|---|
| 100% | 0.5816 |
| 97% | 0.5914 |
| 90% | 0.6166 |
| 80% | 0.6546 |
| 70% | 0.6940 |
| 50% | 0.7624 |

| subconcept | accuracy | n |
|---|---|---|
| Device | 0.5446 | 437 |
| DiagnosisCode | 0.6689 | 2700 |
| EncounterType | 0.5908 | 567 |
| LabService | 0.5664 | 858 |
| MedicationProfileMedication | 0.0000 | 1 |
| ProcedureCode | 0.5056 | 2777 |

Sample errors:


**laterality**

- `lower extremity problem` → gold `451431000124104` Disorder of left lower extremity | pred `724632007` Problem of lower limb
- `primary osteoarthritis of left kneey` → gold `239862000` Idiopathic osteoarthritis | pred `323301000119109` Osteoarthritis of left knee joint
- `right - levator scapula syndrome` → gold `62647006` Psychogenic anal spasm | pred `310401000119102` Impingement syndrome of right shoulder region
- `lvad dependence` → gold `723438005` Left ventricular assist device present | pred `788900007` Dependence on artificial heart
- `hysterectomy w/ salpingo-oopherectomy` → gold `236886002` Hysterectomy | pred `116144002` Total abdominal hysterectomy with bilateral salpingo-oophorectomy
- `acute (rca) anterior st elevation` → gold `15713121000119105` Acute ST segment elevation myocardial infarction due to right coronary artery occlusion | pred `703164000` Acute anterior ST segment elevation myocardial infarction

**retrieval_miss**

- `pt heel slides` → gold `386294003` Joint mobility exercises | pred `256235009` Exercise
- `poct chem/h&h/abg` → gold `386521001` Bedside laboratory testing | pred `405262001` Point of care testing
- `postural change` → gold `258144009` Position change | pred `103017008` Postural dizziness
- `swimming unsupervised` → gold `423562005` Lacks necessary supervision | pred `20461001` Swimming
- `inhalers` → gold `243132000` Inhaled drug administration | pred `334980009` Inhaler
- `ac` → gold `260678004` Use of anticoagulation | pred `44868003` Acupuncture

**semantic-type mismatch (e.g. finding vs procedure)**

- `se` → gold `85097005` Secondary diagnosis | pred `37188009` Selenium measurement
- `brushed her teeth` → gold `470391000124100` Independent brushing of own teeth | pred `450809006` Brushing of teeth
- `referring provider` → gold `397663001` Referral source | pred `3457005` Patient referral
- `exudate` → gold `247450001` Exudate | pred `42142007` Exudate
- `nursing notes` → gold `5531000179105` Nursing report | pred `385778002` Nursing status report
- `smoking cigarettes` → gold `77176002` Smoking | pred `66562002` Cigarette smoking tobacco

**wrong concept, same semantic type**

- `gauze` → gold `334491007` Gauze swab | pred `350785008` Gauzes
- `type 2 diabetes mellitus with other specified` → gold `74627003` Diabetic complication | pred `44054006` Type 2 diabetes mellitus
- `mobility training` → gold `359665005` Ambulation training | pred `118629009` Training
- `prexisting` → gold `102478008` Pre-existing condition | pred `70693003` At risk for impaired skin integrity
- `diabetes in his eyes` → gold `25093002` Disorder of eye due to diabetes mellitus | pred `73211009` Diabetes mellitus
- `doesn't feel any better eating well` → gold `367391008` Malaise | pred `249473004` Altered appetite

## Label consistency (empirical ceiling)

Pairs of training terms that are the same after normalisation (case, punctuation, plural, word order), or nearly identical (char 3-4-gram cosine >= 0.95), and how often their gold codes agree:

| system | canonical-duplicate pairs | same gold code | near-identical pairs (cos>=0.95) | same gold code |
|---|---|---|---|---|
| ICD10CM | 53 | 0.925 | 79 | 0.937 |
| SNOMEDCT_US | 250 | 0.668 | 256 | 0.758 |