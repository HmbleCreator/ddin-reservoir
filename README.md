# DDIN-Reservoir: Navigation Guide

> A spiking neural network research program exploring phonosemantic structure in Sanskrit verbal roots through neuromorphic computing.

---

## 📁 Repository Structure

```
DDIN-Reservoir/
|              
├── Experiments/         # All 57+ experiment scripts
├── SampleData/          # Benchmark datasets
├── data.txt             # data used
├── .gitignore           # This file
└──ddin_agi_research_log.md  # Complete research timeline
```

---

## 📖 Papers (in Papers/)

| Paper                                                                                 | Title                   | Status | Key Finding                                                                 |
| ------------------------------------------------------------------------------------- | ----------------------- | ------ | --------------------------------------------------------------------------- |
| **Paper 1** ([DOI: 10.5281/zenodo.19508957](https://doi.org/10.5281/zenodo.19508957)) | Phonosemantic Grounding | ✅      | Framework establishing Sanskrit as motivated sign system                    |
| **Paper 2** ([DOI: 10.5281/zenodo.19570074](https://doi.org/10.5281/zenodo.19570074)) | Sequential ODE Encoding | ✅ Note | Sequential encoding improves consonant class discrimination                 |
| **Paper 3** ([DOI: 10.5281/zenodo.19602054](https://doi.org/10.5281/zenodo.19602054)) | ESL + BCM Architecture  | ✅      | Epileptiform Synchrony Limit characterization; BCM resolves it              |
| **Paper 4**                                                                           | Negative Result         | ✅      | Acoustic features insufficient for semantic recovery; mechanistic diagnosis |
| **Paper 5**                                                                           | (See Paper 4)           | —      | Paper 4 was retracted and replaced with negative result paper               |


**Critical Reading Order:**
1. Start with **Paper 4** (the negative result) to understand what was discovered
2. Read **Paper 3** for the valid contributions (ESL + BCM)
3. Read **Paper 1** for the theoretical framework

---

## 🧪 Experiments (in Experiments/)

### Key Validated Experiments

| File | Phase | ARI | Description |
|------|-------|-----|-----------|
| `ddin_exp56_locus_taxonomy.py` | 13 | **0.0494** | Reservoir recovers articulatory locus from acoustics |
| `ddin_exp57_semantic_locus_correlation.py` | 13 | **0.0095** | Semantic & locus taxonomies are orthogonal |
| `ddin_exp55_coherence_reset.py` | 12 | **0.2232** | Coherence causal mechanism test |
| `ddin_exp47_vaikharī_decoder.py` | 11 | N/A | Generative decoder for Vaikharī layer |
| `ddin_exp40_mega_scaling_snn.py` | 10 | N/A | Full 2000-root run (before label audit) |
| `ddin_exp45b_5seed_gpu_sweep.py` | 11 | **0.9555** | GPU replication (before label audit) |

### Phase Organization

| Phase | Experiments | Focus |
|-------|-------------|-------|
| **1-4** | exp01-exp14 | Early framework validation |
| **5-8** | exp15-exp21 | Static embedding ceiling, SNN pivot |
| **9-10** | exp22-exp41 | GRPO optimization, Semantic Pressure discovery |
| **11** | exp42-exp53 | GPU replication, Piṅgala integration, Level 2 |
| **12** | exp54-exp55 | Vaikharī generation, Coherence reset |
| **13** | exp56-exp57 | Locus taxonomy validation (mechanistic diagnosis) |

### Naming Convention
- `exp##_description.py` - Main experiment script
- `exp##_variant.py` - Modified version
- All scripts are numbered sequentially

---

## 📊 SampleData/

Benchmark datasets in CSV format:

- `task1_axis_prediction.csv` - 150-root semantic axis benchmark
- `task2_phonological_siblings.csv` - Same-locus pairs
- `task3_fabricated_roots.csv` - Alien roots for zero-shot test
- `task4_cross_locus_distance.csv` - Cross-locus distances
- `task5_rule_generalization.csv` - Held-out generalization
- `task6_trajectories.csv` - Sequential encoding trajectories
- `task7_triplets.csv` - Triplet ordering
- `task8_phonation.csv` - Voicing feature discrimination

---

## 🔬 Key Scientific Findings

### What WAS Validated (Positive Results)
1. **BCM Homeostasis** - Resolves epileptiform synchrony in spiking networks
2. **Semantic Pressure** - Dense corpus (N=2000) induces attractor fusion
3. **Locus Recovery** - Reservoir extracts articulatory place-class from formants (ARI=0.0494)
4. **Sequential Encoding** - Sequential ODE improves over static baseline for consonant discrimination

### What Was NOT Validated (Corrected)
1. **Original ARI = 0.9758** - Label leakage (axis derived from root's initial consonant)
2. **Speed-1 ARI = 0.204** - Same label leakage issue
3. **Acoustic → Semantic mapping** - Semantic/locus taxonomies are near-orthogonal (ARI=0.0095)

### Mechanistic Diagnosis (from Paper 4)
The reservoir **can** process acoustic information (locus at ARI=0.0494). The failure to recover semantic structure stems from the semantic taxonomy being nearly orthogonal to the articulatory features that acoustic formants encode.

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/HmbleCreator/ddin-reservoir.git

# Navigate to experiments
cd DDIN-Reservoir/Experiments/

# Run a key experiment (requires PyTorch + GPU)
python ddin_exp56_locus_taxonomy.py
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy, Scikit-learn
- GPU (recommended for spiking network simulations)

---

## 📋 Research Log

The complete timeline is in `ddin_agi_research_log.md` (2000+ lines).

Key milestones:
- **April 2025**: Project inception
- **April 2026**: Semantic Pressure discovery (ARI=0.9758)
- **April 2026**: Label audit identifies label leakage
- **April 2026**: Phase 13 mechanistic diagnosis

---

## 🤝 Contributing

This is an independent research project. All code is provided as-is for reproducibility.

---

## 📄 License

CC BY 4.0 - See LICENSE file for details.

---

*Last updated: April 2026*
