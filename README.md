# Dere Detector: Multimodal Archetype Classification

Task #3 Seleksi Laboratorium Intelegensi Buatan 2026  
Riantama Putra, 18224061

This repository contains the complete solution for **Dere Detector**, a multimodal classification task for predicting three character archetypes:

- `deredere`
- `kuudere`
- `tsundere`

The solution combines **personality text, character name, and image information**. The final selected model is **V3.2 + EVA×2**, a constrained late-fusion ensemble that achieved a **Public Macro F1 of 0.83504**.

## Repository Structure

```text
Task3_AI_18224061/
├── src/
│   └── inference.py
├── notebooks/
│   └── dere.ipynb
├── docs/
│   └── Task3_AI_18224061.pdf
└── README.md
```

- `notebooks/dere.ipynb`: complete EDA, preprocessing, validation, modeling, experiments, error analysis, explainability, real-world testing, and final submission pipeline.
- `src/inference.py`: standalone inference script for the saved final model.
- `docs/Task3_AI_18224061.pdf`: answers for Skyfall, Vesper, and Spectre.
- `README.md`: repository overview and execution instructions.

## Final Solution

The final model uses five complementary branches:

| Branch | Role |
| --- | --- |
| DeBERTa-v3 NLI | Primary semantic text expert |
| ModernBERT | Independent text representation |
| TF-IDF | Classical lexical text signal |
| EVA02 | Anime-specific visual representation and visual cues |
| SigLIP2 | General visual representation |

The final prediction is produced using **constrained late fusion**. The selected **EVA×2** variant doubles the contribution of the EVA02 branch before the fusion weights are normalized.

The model is intentionally text-dominant because personality descriptions provide the strongest semantic signal, while visual branches contribute complementary information on ambiguous cases.

### Key Results

| Experiment | Result |
| --- | ---: |
| TF-IDF + ConvNeXt, zero-Transformer baseline OOF Macro F1 | 0.62253 |
| DeBERTa-v3 NLI OOF Macro F1 | ~0.7495 |
| EVA02 standalone OOF Macro F1 | 0.56179 |
| V3.2 + EVA×2 canonical OOF Macro F1 | 0.75816 |
| V3.2 + EVA×2 Public Macro F1 | **0.83504** |

The experiments also showed that a higher offline score did not always translate to a better leaderboard result. Several later experiments improved OOF performance but generalized worse on the public leaderboard. This was the main reason V3.2 + EVA×2 was retained as the final model.

## Modeling Notes

The notebook includes both required architecture families:

1. **Zero-Transformer pipeline**  
   Word and character TF-IDF for text combined with a WD ConvNeXt V3 CNN visual branch.

2. **Transformer-based multimodal pipeline**  
   DeBERTa-v3 NLI and ModernBERT for text, combined with EVA02 and SigLIP2 visual representations.

Important data-handling decisions include:

- `file_name` is used only for image I/O because its folder prefix leaks the training label.
- The label space is fixed to `deredere`, `kuudere`, and `tsundere`.
- The `dandere` value found in `sample_submission.csv` is not used to determine the label space.
- Validation uses a fixed stratified 5-fold split.
- No external competition data is used.
- Pretrained models are used only as permitted non-generative encoders/classifiers.
- No generative LLM or generative VLM is used for prediction.

## Saved Final Model

The saved final model can be downloaded from Google Drive:

**[Download DereDetector Final Model](https://drive.google.com/drive/folders/1BAdcZLKLBRF3aubXMwKJu9M4wdyySAm4?usp=sharing)**

Expected model bundle:

```text
DereDetector_Final_Model/
├── models_v3/
│   ├── nli_large_full_full/
│   ├── modernbert_full/
│   ├── tfidf_full.joblib
│   ├── eva_full_head.joblib
│   └── siglip2_full_head.joblib
└── final_config_v3.json
```

`nli_large_full_full/` and `modernbert_full/` contain the fine-tuned Transformer weights. The EVA02 and SigLIP2 files contain the trained downstream classification heads, while their pretrained backbones are loaded from Hugging Face during inference.

## Inference

### 1. Prepare the Environment

Python 3.12 is recommended to stay close to the training environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the inference dependencies:

```bash
python -m pip install \
  joblib \
  numpy \
  pillow \
  "scikit-learn==1.6.1" \
  torch \
  torchvision \
  "transformers>=4.56,<5" \
  timm \
  huggingface_hub \
  safetensors \
  sentencepiece
```

Using `scikit-learn==1.6.1` is recommended because the saved scikit-learn estimators were serialized with that version.

### 2. Download the Saved Model

Download the Google Drive folder and place it anywhere accessible to the inference script.

For example:

```text
Task3_AI_18224061/
├── DereDetector_Final_Model/
├── src/
│   └── inference.py
├── notebooks/
│   └── dere.ipynb
├── docs/
│   └── Task3_AI_18224061.pdf
└── README.md
```

### 3. Validate the Model Bundle

From the repository root:

```bash
python src/inference.py \
  --model-dir DereDetector_Final_Model \
  --check-only
```

A valid bundle should report the three labels and the five final branches.

### 4. Run Inference

```bash
python src/inference.py \
  --model-dir DereDetector_Final_Model \
  --image path/to/image.png \
  --name "Character Name" \
  --personality "Personality description."
```

For longer personality descriptions, a text file can be used:

```bash
python src/inference.py \
  --model-dir DereDetector_Final_Model \
  --image path/to/image.png \
  --name "Character Name" \
  --personality-file path/to/personality.txt
```

To save the complete prediction output:

```bash
python src/inference.py \
  --model-dir DereDetector_Final_Model \
  --image path/to/image.png \
  --name "Character Name" \
  --personality-file path/to/personality.txt \
  --output-json prediction.json
```

Example output:

```text
Prediction : deredere
Probabilities:
  deredere  0.697277
  kuudere   0.243136
  tsundere  0.059588

Visual cues: looking at viewer, closed mouth, smile, half-closed eyes, light smile
Branch predictions:
  nli_large    deredere
  modernbert   deredere
  tfidf        deredere
  eva02        kuudere
  siglip2      tsundere
```

The first full inference requires an internet connection because the pretrained EVA02 and SigLIP2 backbones are loaded from Hugging Face.

## Running the Notebook

The complete modeling workflow is available in:

```text
notebooks/dere.ipynb
```

The notebook covers:

- problem understanding and solution hypotheses
- exploratory data analysis
- leakage and duplicate auditing
- preprocessing and feature engineering
- zero-Transformer baseline
- Transformer-based text and visual modeling
- architecture comparison
- multimodal fusion
- final architecture selection
- error analysis
- explainability analysis
- real-world testing
- final submission generation
- saved-model export

For full training, a Kaggle GPU runtime is recommended.

## Bonus Presentation

The bonus presentation explaining the multimodal solution is available on YouTube:

**[Watch the Dere Detector Presentation](https://youtu.be/C3USpEZVawQ)**

## Deliverables

| Deliverable | Location |
| --- | --- |
| Main notebook | [`notebooks/dere.ipynb`](notebooks/dere.ipynb) |
| PDF answers | [`docs/Task3_AI_18224061.pdf`](docs/Task3_AI_18224061.pdf) |
| Standalone inference | [`src/inference.py`](src/inference.py) |
| Saved final model | [Google Drive](https://drive.google.com/drive/folders/1BAdcZLKLBRF3aubXMwKJu9M4wdyySAm4?usp=sharing) |
| Bonus presentation | [YouTube](https://youtu.be/C3USpEZVawQ) |

## Author

**Riantama Putra**  
18224061
