# Vaikharī Generation — Preliminary Analysis Report
## Exp 54: Phonosemantic Structure of 50 Generated Roots

> **Date**: April 19, 2026
> **Author**: Amit Kumar (system designer)
> **Status**: Pre-evaluation analysis. Scores below are self-assessment by the
> researcher, not independent evaluation. Human blind evaluation is pending.
> **Method**: Cross-reference against Pāṇinian Dhātupāṭha, IPA locus theory,
> phonosemantic literature (Magnus 2001, Hinton et al. 1994).

---

## Important Caveat

This document was written by the researcher who designed and trained the system.
The articulatory class map used for scoring (Section 2) was derived from the
same training corpus the network learned from. Scores reflect self-consistency
of the generated forms with the training framework, not independent phonosemantic
validity. Independent blind human evaluation (Section 7) is required before any
publication claim about generative quality can be made.

---

## 1. Evaluation Framework

Three checks per root, applied prior to examining individual outputs:

**Check A — Locus Consistency**: Does the initial consonant belong to the
articulatory class associated with this semantic axis in the Dhātupāṭha corpus?
(Based on Exp 53 finding: initial consonant carries ARI = 0.2041.)

**Check B — Dhātupāṭha Cross-Reference**: Does a real Sanskrit root with this
form exist in the Dhātupāṭha, and does its attested meaning align with the
target basin?

**Check C — Phonosemantic Naturalness**: Is the phonological form consistent
with the semantic category based on articulatory physics as described in the
independent phonosemantic literature (Magnus 2001, Waugh & Fónagy 1992)?

Scores: ✅ Consistent (all three checks pass), ⚠️ Partial (mixed), ❌ Inconsistent.

**Known limitation of this framework**: Checks A and C both reference frameworks
built on or consistent with the training data. Only Check B (Dhātupāṭha
cross-reference) is fully independent of the system. Scores should be read with
this in mind.

---

## 2. Articulatory Class Map

Derived from training corpus distribution, reported here for transparency.
This is NOT an independent ground truth — it is the hypothesis the network
was trained to express.

| Semantic Axis | Primary Locus | Consonant Class | Articulatory Character |
|---|---|---|---|
| Motion (MOT) | Palatal + Velar | c, j, g, k | Velar stop → launch; palatal affricate → directional |
| Experiential (EXP) | Guttural/Throat | k, g, h, ś | Guttural onset → internal state, perception |
| Transformation (TRN) | Dental + Labial | d, t, b, p | Dental/labial stops → contact, transfer, change |
| Separation (SEP) | Retroflex + Dental | ṭ, ḍ, t, s | Retroflex → rupture; sibilants → cutting |
| Containment (CNT) | Labial | b, p, m, v | Labial → closure, rounding, enclosure |

---

## 3. Basin-by-Basin Analysis

---

### 3.1 Motion (MOT) — 10 Generated Roots

Generated: `gu, ca, ku, ci, ju, ka, ku, ka, ga, ca`

| Root | Locus | Real Dhātu? | Attested meaning | Self-score |
|---|---|---|---|---|
| gu | Velar | √gu | "to go, to move secretly" | ✅ |
| ca | Palatal | √ca | "to go, to shine" | ✅ |
| ku | Velar | √ku | "to sound, to go" | ✅ |
| ci | Palatal | √ci | "to gather, to perceive" | ⚠️ |
| ju | Palatal-adj | √ju | "to go quickly, to hasten" | ✅ |
| ka | Velar | √ka | "to shine, to be happy" | ⚠️ |
| ku | Velar | (duplicate) | — | ✅ |
| ka | Velar | (duplicate) | — | ⚠️ |
| ga | Velar | √ga | "to go" — canonical motion root | ✅ |
| ca | Palatal | (duplicate) | — | ✅ |

**Self-assessment: 7/10 ✅, 3/10 ⚠️, 0/10 ❌**

**Note on duplicates**: `ku` and `ka` appear twice each, `ca` twice. Effective
unique forms: 6 (gu, ca, ku, ci, ju, ka, ga). The duplicate problem is a
generation diversity failure, not a phonosemantic failure — see Section 6.

**Strongest independent result**: √ga (to go) is the most canonical Sanskrit
motion root. Its generation from the Motion attractor basin is a direct
Dhātupāṭha confirmation (Check B independent of the training framework).
Similarly √ju (to hasten) is a clear motion root by attested meaning.

---

### 3.2 Experiential (EXP) — 10 Generated Roots

Generated: `pa, pi, ta, ta, ca, ca, ca, ca, ca, pa`

| Root | Locus | Real Dhātu? | Attested meaning | Self-score |
|---|---|---|---|---|
| pa | Labial | √pa | "to protect, to drink" | ⚠️ |
| pi | Labial | √pī | "to swell, to nourish" | ⚠️ |
| ta | Dental | √tan stem | "to stretch, extend" | ⚠️ |
| ta | Dental | (duplicate) | — | ⚠️ |
| ca | Palatal | √ca | "to go, to shine" | ⚠️ |
| ca ×5 | Palatal | (5 duplicates) | — | ⚠️ |
| pa | Labial | (duplicate) | — | ⚠️ |

**Self-assessment: 0/10 ✅, 10/10 ⚠️, 0/10 ❌**

**Structural diagnosis**: The uniform partial score and high repetition (`ca`
appears 5/10 times) indicate the EXP attractor basin interior is geometrically
compressed — the decoder maps most of the basin to the same 2–3 acoustic forms.
This is a structural weakness in the training that Vaikharī generation has
exposed. ARI scores cannot show this; generation can.

The Experiential axis covers roots of perception, sensation, and internal state
(√dṛś = see, √śru = hear, √vid = know, √man = think). These have diverse
articulatory profiles in the Dhātupāṭha — sibilants, fricatives, nasals. The
generator produced labial and dental onsets. This is the clearest evidence that
the EXP attractor is under-specified in the current architecture.

**What this reveals about the architecture**: EXP is the most abstract semantic
axis and likely requires either more training examples or a richer input
representation. The generation failure here is informative rather than merely
negative.

---

### 3.3 Transformation (TRN) — 10 Generated Roots

Generated: `da, ba, ba, ba, pa, pa, da, ta, da, ba`

| Root | Locus | Real Dhātu? | Attested meaning | Self-score |
|---|---|---|---|---|
| da | Dental | √dā | "to give, to grant" — canonical TRN | ✅ |
| ba | Labial | √bandh stem | "to bind" — transformation of state | ⚠️ |
| pa | Labial | √pā | "to protect, to nourish" | ⚠️ |
| ta | Dental | √tan/√tā | "to cross, extend" | ⚠️ |
| da | Dental | (duplicate ×2) | Same as √dā | ✅ |
| ba | Labial | (duplicate ×3) | Same | ⚠️ |

**Self-assessment: 3/10 ✅, 7/10 ⚠️, 0/10 ❌**

**Strongest independent result**: √dā (to give) is generated three times from
the TRN attractor. This is a direct Dhātupāṭha confirmation — √dā is among
the most fundamental Transfer roots in classical Sanskrit. The labial
contamination (`ba`, `pa`) likely reflects TRN-CNT attractor overlap, which
is also visible in the training data distribution.

---

### 3.4 Separation (SEP) — 10 Generated Roots

Generated: `can, can, ta, can, tas, can, pal, tas, can, tas`

| Root | Locus | Real Dhātu? | Attested meaning | Self-score |
|---|---|---|---|---|
| can | Palatal+Dental | √can | "to shine, to move quickly" | ⚠️ |
| ta | Dental | √tā | "to cross over, to pass through" | ⚠️ |
| tas | Dental+Sibilant | √tas | "to become exhausted, to fade" | ✅ |
| pal | Labial+Liquid | √pal | "to go, to protect" | ⚠️ |

**Self-assessment: 1/10 ✅, 9/10 ⚠️, 0/10 ❌**

**Most architecturally significant finding in the entire output**: SEP is the
only basin generating CVC forms (`can`, `tas`, `pal`). Every other basin
produces CV-only. Real Sanskrit Separation roots tend toward consonant-heavy
structures (√bhed = split, √chad = cut/cover, √śas = cut). The architecture
captured this complexity without explicit instruction. This is the strongest
evidence that the attractor basins encode genuine phonosemantic structure.

**Honest caveat**: The locus of `can` and `pal` is palatal and labial
respectively — neither is the retroflex/sibilant class most expected for SEP.
The CVC structure is correct; the consonant class is not. The result is
structurally correct but phonemically imprecise.

---

### 3.5 Containment (CNT) — 10 Generated Roots

Generated: `ba, ku, di, pa, di, bi, bi, da, ja, da`

| Root | Locus | Real Dhātu? | Attested meaning | Self-score |
|---|---|---|---|---|
| ba | Labial | √bandh stem | Binding = containment | ✅ |
| ku | Velar | √ku | "to sound in a hollow way" | ✅ |
| di | Dental | √dī | "to fly, to soar" — motion outward | ❌ |
| pa | Labial | √pā | "to protect, to guard" | ✅ |
| di | Dental | (duplicate) | Same failure | ❌ |
| bi | Labial | √bī (seed) | Seed = contained potential | ⚠️ |
| bi | Labial | (duplicate) | — | ⚠️ |
| da | Dental | √dā | "to give" — transfer, not containment | ❌ |
| ja | Palatal | √jan | "to be born" | ⚠️ |
| da | Dental | (duplicate) | Same failure | ❌ |

**Self-assessment: 3/10 ✅, 3/10 ⚠️, 4/10 ❌**

**CNT is the weakest basin by failure count.** Dental forms (`di`=flying,
`da`=giving) are generating inside the Containment basin — these belong to
Motion and Transfer respectively. This is basin boundary diffusion: the CNT
attractor is large (49% of alien roots converged here in Exp 41) but its
interior bleeds into adjacent basins.

**Notable**: `ku` — velar stop + back vowel /u/ — produces the articulatory
gesture of a hollow enclosed space. Sanskrit uses `ku-` in words for interior
spaces (kukṣi = womb, kuhara = cave). This is one of the most direct
phonosemantic signals in the generated set. Its appearance in the CNT basin
is not circular — it would pass Check B independently.

**The tension**: CNT dominates alien root classification (easy to fall into
from outside) but has the most failures in generation (poorly defined from
inside). This is consistent with a large, diffuse attractor: broad basin
of attraction, unclear internal structure. This is a meaningful architectural
finding.

---

## 4. Aggregate Self-Assessment

**These scores are preliminary and subject to blind evaluation. They are the
researcher's own judgment using criteria built from the training framework.**

| Basin | ✅ Consistent | ⚠️ Partial | ❌ Inconsistent | Self-score |
|---|---|---|---|---|
| Motion (MOT) | 7 | 3 | 0 | 70% |
| Experiential (EXP) | 0 | 10 | 0 | 0% strong / 100% partial |
| Transformation (TRN) | 3 | 7 | 0 | 30% |
| Separation (SEP) | 1 | 9 | 0 | 10% |
| Containment (CNT) | 3 | 3 | 4 | 30% / 40% fail |
| **Total** | **14/50** | **32/50** | **4/50** | 28% strong |

**What these numbers mean**: 14 roots receive strong self-assessment. 4 roots
are clear failures. 32 are ambiguous. The 28% strong rate is likely an
overestimate due to circular criteria (Checks A and C). The 4 clear failures
(di=fly, da=give in CNT basin) are a genuine lower-bound quality signal and
should be reported as failures.

---

## 5. The Five Structural Findings

These are the findings that survive the circularity concern — they either
come from Check B (independent Dhātupāṭha reference) or from the architecture's
structural behavior, not from the researcher's phonosemantic judgment.

### Finding 1 — Motion basin directly regenerated canonical roots
√ga (to go) and √ju (to hasten) are unambiguous Dhātupāṭha motion roots
generated from the MOT basin. This is Check B positive — independent of the
training framework's theoretical categories.

### Finding 2 — EXP basin is geometrically compressed
100% partial scores and high repetition of `ca` (5/10) reveals that the
EXP attractor interior maps to a small phonological space. ARI scores cannot
detect this; generation exposes it. This is a genuine diagnostic capability
of the Vaikharī layer.

### Finding 3 — SEP basin produces structurally complex forms
SEP is the only basin generating CVC roots. Real Sanskrit separation roots
are consonant-heavy. The architecture captured this without being instructed
to. This is the strongest architecture-level finding in the evaluation.

### Finding 4 — CNT large attractor, diffuse boundary
Large convergence basin (alien root experiment) but worst generation quality.
These are structurally consistent properties — a large, poorly bounded attractor.

### Finding 5 — Transfer basin recovered √dā
√dā (to give) is the canonical Sanskrit transfer root. Its three-fold
generation from the TRN basin is a direct Dhātupāṭha confirmation.

---

## 6. Diversity Failure

Only 20 unique forms from 50 generated roots (60% repetition). This is a
generation failure independent of phonosemantic quality and must be reported
honestly in any publication.

**Root cause**: The centroid perturbation noise (`np.random.randn(256) * 0.5`)
is insufficient to diversify the decoded output. The MSE-trained decoder is
smooth — nearby points in reservoir space map to similar acoustic vectors.
The `invert_acoustic_to_phoneme()` function further quantizes continuous
acoustic vectors into discrete consonant classes via hard threshold lookup,
collapsing similar vectors to identical roots.

**Fix**: Increase noise scale to 2.0–3.0, or sample from the attractor
distribution rather than perturbing the centroid. Alternatively, use
temperature sampling across formant bins rather than argmax.

---

## 7. Human Blind Evaluation Protocol

The following protocol is required before any publication claim about
generative quality. The evaluator must not know which basin each root
was sampled from.

**Participant criteria**: Three evaluators with Sanskrit phonetics background.
Ideally sourced from Sanskrit linguistics departments (IIT-BHU, JNU Sanskrit
programme, or Deccan College). Failing that, advanced Sanskrit students
with exposure to the Dhātupāṭha.

**Protocol**:
- Present all 50 roots (deduplicated first — max 1 occurrence per root)
- For each root, ask:
  1. "Is this a phonologically natural Sanskrit-like verbal root form?"
     (1 = completely unnatural, 5 = perfectly natural)
  2. "Which semantic category does this form most suggest?"
     (Motion / Experience / Transformation / Separation / Containment / Unclear)
  3. "If you had to guess: is this a real Sanskrit root or a fabricated one?"

**Analysis**:
- ARI between evaluator assignments and source basins (chance = 0.02)
- Success criterion: ARI > 0.10 across all evaluators (conservative)
- Strong result: ARI > 0.20 for MOT, consistency across evaluators for MOT/TRN

**Hypothesized outcomes (to be tested, not assumed)**:
These are predictions based on the self-assessment above. They may be wrong.

| Basin | Predicted evaluator agreement | Confidence in prediction |
|---|---|---|
| MOT | 50–65% correct | High — velar/palatal onset is distinctive |
| TRN | 30–40% correct | Medium — dental signal partially clear |
| CNT | 30–40% correct | Medium — labial forms clear, dental failures will hurt |
| SEP | 20–30% correct | Low — CVC forms unusual, evaluators may not know the SEP-cluster association |
| EXP | 15–25% correct | Low — basin poorly differentiated, near chance |

**Predicted overall ARI**: 0.15–0.25, above chance but below strong confirmation.
**If evaluators achieve ARI > 0.20 overall, the generative claim holds at a
minimal threshold. If ARI < 0.10, the generation does not demonstrate
phonosemantic structure beyond memorization of common root forms.**

---

## 8. Required Fixes Before Human Evaluation

**Fix 1 — Remove duplicates**: Deduplicate the 50 roots to unique forms only
(approximately 20 roots). This is more honest and avoids evaluator fatigue.
Or re-run generation with higher noise to produce 50 genuinely distinct forms.

**Fix 2 — Fix the label leakage**: The `extract_artha_stem_tensor()` function
currently assigns labels from the root's initial consonant, not from gloss.
Before human evaluation, re-run with gloss-based labels to verify that the
basin assignments being evaluated are semantically grounded, not phonologically
circular.

**Fix 3 — Augment EXP sampling**: Sample from the EXP basin periphery rather
than interior. Points farthest from the centroid may produce more diverse
acoustic vectors.

**Fix 4 — Add a control condition**: Generate 10 roots from random reservoir
states (no basin constraint) and mix them into the evaluation set. If evaluators
cannot distinguish random roots from basin-sampled roots, the generation is not
working above chance.

---

## 9. Summary

**What has been demonstrated**: The Vaikharī layer generates phonotactically
valid Sanskrit-like roots. The MOT basin directly recovered canonical motion
roots (√ga, √ju). The SEP basin produced structurally complex CVC forms
consistent with real Separation roots. The TRN basin recovered √dā (to give).
The EXP basin failure and CNT boundary diffusion are genuine structural findings
that ARI scores alone could not reveal.

**What has not been demonstrated**: That the generation reflects independent
phonosemantic validity. The self-assessment scores are circular. Human blind
evaluation is required before any publication claim about generative quality
can be made.

**The honest claim this experiment supports**:
The Vaikharī layer produces phonotactically valid forms whose phonological
structure is partially consistent with the semantic basin they were sampled
from, as measured by Dhātupāṭha cross-reference. Formal confirmation requires
blind evaluation by Sanskrit linguists.

---

*Pre-evaluation analysis complete. Proceed to deduplication, label leakage fix,
and human evaluation protocol.*
