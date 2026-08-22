#!/usr/bin/env python3
"""Standalone inference for the final Dere Detector V3.2 EVA×2 ensemble.

This script mirrors the saved-model inference path used in the final
`dere.ipynb` notebook. It loads the fine-tuned/saved heads from Google Drive
and downloads only the two public pretrained visual backbones required by the
final ensemble:

- SmilingWolf/wd-eva02-large-tagger-v3
- google/siglip2-base-patch16-384

Expected model bundle
---------------------
DereDetector_Final_Model/
├── models_v3/
│   ├── nli_large_full_full/
│   ├── modernbert_full/
│   ├── tfidf_full.joblib
│   ├── eva_full_head.joblib
│   └── siglip2_full_head.joblib
└── final_config_v3.json

Example
-------
python src/inference.py \
  --model-dir /path/to/DereDetector_Final_Model \
  --image /path/to/character.png \
  --name "Character Name" \
  --personality "A cheerful and affectionate character."

Required packages
-----------------
numpy, pillow, joblib, scikit-learn, torch, transformers, timm,
huggingface_hub, sentencepiece

Internet access is required the first time EVA02/SigLIP2 (and optionally the
original NLI tokenizer) are loaded from Hugging Face.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
from PIL import Image


EXPECTED_BRANCHES = ["nli_large", "modernbert", "tfidf", "eva02", "siglip2"]
HYP_TEMPLATES = [
    "This character fits the {cls} archetype; the training data associates it with traits such as {traits}.",
    "The described personality is consistent with {cls}, with signals such as {traits}.",
    "This is a {cls} character. Relevant personality cues include {traits}.",
]
EMOTION_TAG_WHITELIST = {
    "smile", "grin", "smirk", "blush", "frown", "expressionless", "serious",
    "angry", "annoyed", "pout", "crying", "tears", "sad", "happy", "laughing",
    "open_mouth", "closed_mouth", "closed_eyes", "half-closed_eyes", "looking_away",
    "looking_at_viewer", "embarrassed", "shy", "scared", "surprised", "confused",
    "nervous", "sweat", "sweatdrop", "furrowed_brow", "raised_eyebrow",
    "clenched_teeth", "teeth", "tongue", "wince", "sleepy", "bored", "emotionless",
    "evil_smile", "light_smile", "wide-eyed",
}


# -----------------------------------------------------------------------------
# Pure utilities. These intentionally match the final notebook implementation.
# -----------------------------------------------------------------------------

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(image, ((s - w) // 2, (s - h) // 2))
    return canvas


def visual_cue_strings(
    tags: np.ndarray,
    names: Sequence[str],
    topn: int = 5,
    threshold: float = 0.20,
) -> np.ndarray:
    norm = [str(x).strip().lower() for x in names]
    idx = [i for i, n in enumerate(norm) if n in EMOTION_TAG_WHITELIST]
    if not idx:
        return np.asarray([""] * tags.shape[0], dtype=object)

    arr = np.asarray(tags)[:, idx]
    nm = np.asarray([norm[i] for i in idx], dtype=object)
    out: List[str] = []
    for row in arr:
        order = np.argsort(row)[::-1]
        chosen: List[str] = []
        for j in order:
            if row[j] < threshold:
                break
            chosen.append(str(nm[j]).replace("_", " "))
            if len(chosen) >= topn:
                break
        out.append(", ".join(chosen))
    return np.asarray(out, dtype=object)


def make_hypothesis(cls: str, traits: Sequence[str], template_id: int = 0) -> str:
    return HYP_TEMPLATES[template_id % len(HYP_TEMPLATES)].format(
        cls=cls,
        traits=", ".join(traits),
    )


def make_premise(name: str, personality: str, visual_cues: str = "") -> str:
    s = f"Character: {name}. Personality: {personality}"
    if visual_cues:
        s += f" Visual cues from the character image: {visual_cues}."
    return s


def _align_classifier_probabilities(
    classifier: Any,
    probabilities: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    label2id = {label: i for i, label in enumerate(labels)}
    out = np.zeros((len(probabilities), len(labels)), dtype=np.float64)
    for j, cls in enumerate(classifier.classes_):
        key = str(cls)
        if key not in label2id:
            raise ValueError(f"Classifier contains unexpected class {key!r}; expected {list(labels)}")
        out[:, label2id[key]] = probabilities[:, j]
    return out


def _aligned_linear_scores(classifier: Any, x: Any, labels: Sequence[str]) -> np.ndarray:
    raw = np.asarray(classifier.decision_function(x))
    if raw.ndim == 1:
        raw = raw[:, None]
    label2id = {label: i for i, label in enumerate(labels)}
    out = np.zeros((x.shape[0], len(labels)), dtype=np.float64)
    for j, cls in enumerate(classifier.classes_):
        key = str(cls)
        if key not in label2id:
            raise ValueError(f"Classifier contains unexpected class {key!r}; expected {list(labels)}")
        out[:, label2id[key]] = raw[:, j]
    return out


def fuse_probabilities(
    branch_probabilities: Mapping[str, np.ndarray],
    core_names: Sequence[str],
    weights: Mapping[str, float],
    bias: Sequence[float],
    labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, str]:
    missing = [name for name in core_names if name not in branch_probabilities]
    if missing:
        raise KeyError(f"Missing branch probabilities: {missing}")

    p = np.zeros(len(labels), dtype=np.float64)
    for name in core_names:
        p += float(weights[name]) * np.asarray(branch_probabilities[name], dtype=np.float64)

    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError(f"Invalid fused probability vector: {p}")

    scores = np.log(np.clip(p, 1e-12, 1.0)) + np.asarray(bias, dtype=np.float64)
    pred = str(labels[int(scores.argmax())])
    return p, scores, pred


# -----------------------------------------------------------------------------
# Model bundle validation / device handling
# -----------------------------------------------------------------------------

def _has_model_weights(directory: Path) -> bool:
    return (directory / "model.safetensors").exists() or (directory / "pytorch_model.bin").exists()


def resolve_bundle_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "final_config_v3.json").exists() and (path / "models_v3").is_dir():
        return path
    if path.name == "models_v3" and (path.parent / "final_config_v3.json").exists():
        return path.parent
    raise FileNotFoundError(
        "Model bundle not found. Expected final_config_v3.json and models_v3/ under "
        f"{path}"
    )


def validate_model_bundle(model_root: Path | str) -> Dict[str, Any]:
    root = resolve_bundle_root(Path(model_root))
    models = root / "models_v3"
    config_path = root / "final_config_v3.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    required_config = [
        "labels", "models", "core_names", "final_eva_x2_weights", "main_bias",
        "temperatures", "stable_traits",
    ]
    missing_cfg = [k for k in required_config if k not in config]
    if missing_cfg:
        raise KeyError(f"final_config_v3.json is missing keys: {missing_cfg}")

    labels = list(config["labels"])
    if labels != ["deredere", "kuudere", "tsundere"]:
        raise ValueError(f"Unexpected label order in config: {labels}")

    core_names = list(config["core_names"])
    if core_names != EXPECTED_BRANCHES:
        raise ValueError(
            "Unexpected core_names. The final V3.2 EVA×2 inference expects "
            f"{EXPECTED_BRANCHES}, got {core_names}"
        )

    weights = config["final_eva_x2_weights"]
    if weights is None or any(name not in weights for name in core_names):
        raise ValueError("final_eva_x2_weights is missing one or more final branches")
    if not np.isclose(sum(float(weights[n]) for n in core_names), 1.0, atol=1e-5):
        raise ValueError(f"Final EVA×2 weights must sum to 1, got {weights}")

    if len(config["main_bias"]) != len(labels):
        raise ValueError("main_bias length does not match label count")

    required_files = [
        models / "tfidf_full.joblib",
        models / "eva_full_head.joblib",
        models / "siglip2_full_head.joblib",
    ]
    for f in required_files:
        if not f.is_file():
            raise FileNotFoundError(f"Missing saved model artifact: {f}")

    for dname in ["nli_large_full_full", "modernbert_full"]:
        d = models / dname
        if not d.is_dir() or not (d / "config.json").exists() or not _has_model_weights(d):
            raise FileNotFoundError(
                f"Missing Hugging Face saved model in {d}. Expected config.json and model weights."
            )

    for key in ["eva", "siglip2", "nli_large", "modernbert"]:
        if key not in config["models"] or not str(config["models"][key]).strip():
            raise KeyError(f"config['models'] is missing {key!r}")

    for label in labels:
        if label not in config["stable_traits"]:
            raise KeyError(f"stable_traits is missing label {label!r}")

    return config


def _select_device(requested: str = "auto") -> str:
    import torch

    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return "cpu"

    # Do a real kernel probe. This catches cases such as a Pascal P100 with a
    # PyTorch/CUDA build that no longer ships kernels for that architecture.
    try:
        x = torch.ones(8, device="cuda")
        y = x * 2
        _ = float(y.sum().cpu())
        del x, y
        torch.cuda.empty_cache()
        return "cuda"
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError(f"CUDA kernel probe failed: {exc}") from exc
        warnings.warn(
            f"CUDA is visible but unusable ({exc}). Falling back to CPU inference.",
            RuntimeWarning,
        )
        return "cpu"


def _amp_context(device: str):
    if device == "cuda":
        import torch
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _clean_torch(device: str) -> None:
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Final predictor
# -----------------------------------------------------------------------------
class DereDetectorPredictor:
    """Exact saved-model inference path for the final V3.2 EVA×2 ensemble."""

    def __init__(self, model_root: Path | str, device: str = "auto") -> None:
        self.root = resolve_bundle_root(Path(model_root))
        self.models_dir = self.root / "models_v3"
        self.config = validate_model_bundle(self.root)
        self.labels: List[str] = list(self.config["labels"])
        self.core_names: List[str] = list(self.config["core_names"])
        self.weights: Dict[str, float] = {
            k: float(v) for k, v in self.config["final_eva_x2_weights"].items()
        }
        self.bias = np.asarray(self.config["main_bias"], dtype=np.float64)
        self.temperatures = {k: float(v) for k, v in self.config["temperatures"].items()}
        self.stable_traits: Dict[str, List[str]] = {
            k: list(v) for k, v in self.config["stable_traits"].items()
        }
        self.model_names = dict(self.config["models"])
        self.device = _select_device(device)

        # The sklearn heads are small and safe to keep resident.
        self.tfidf = joblib.load(self.models_dir / "tfidf_full.joblib")
        self.eva_head = joblib.load(self.models_dir / "eva_full_head.joblib")
        self.siglip_head = joblib.load(self.models_dir / "siglip2_full_head.joblib")

        self._validate_saved_heads()

    def _validate_saved_heads(self) -> None:
        for key in ["vectorizer", "classifier", "temperature"]:
            if key not in self.tfidf:
                raise KeyError(f"tfidf_full.joblib is missing {key!r}")
        for key in ["selector", "scaler", "classifier"]:
            if key not in self.eva_head:
                raise KeyError(f"eva_full_head.joblib is missing {key!r}")
        for key in ["scaler", "classifier", "temperature"]:
            if key not in self.siglip_head:
                raise KeyError(f"siglip2_full_head.joblib is missing {key!r}")

    def _extract_eva(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        import torch
        import timm
        from huggingface_hub import hf_hub_download
        from timm.data import create_transform, resolve_data_config

        repo = str(self.model_names["eva"])
        model = timm.create_model(f"hf_hub:{repo}", pretrained=True).to(self.device).eval()
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        tags_csv = hf_hub_download(repo_id=repo, filename="selected_tags.csv")
        names: List[str] = []
        gen_idx: List[int] = []
        with open(tags_csv, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if int(row["category"]) == 0:
                    gen_idx.append(i)
                    names.append(row["name"])

        with Image.open(image_path) as im:
            x = transform(pil_pad_square(pil_ensure_rgb(im))).unsqueeze(0)
        # WD v3 TIMM convention used by the training notebook: RGB -> BGR.
        x = x[:, [2, 1, 0], :, :].to(self.device)

        with torch.inference_mode(), _amp_context(self.device):
            feat = model.forward_features(x)
            emb = model.forward_head(feat, pre_logits=True)
            logits = model.forward_head(feat, pre_logits=False)

        idx_tensor = torch.as_tensor(gen_idx, device=logits.device, dtype=torch.long)
        tags = torch.sigmoid(logits).index_select(1, idx_tensor).float().cpu().numpy()
        emb_np = emb.float().cpu().numpy()
        del model, x, feat, emb, logits, idx_tensor
        _clean_torch(self.device)
        return tags, emb_np, names

    def _extract_siglip2(self, image_path: Path) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoProcessor

        repo = str(self.model_names["siglip2"])
        processor = AutoProcessor.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).to(self.device).eval()

        with Image.open(image_path) as im:
            image = pil_pad_square(pil_ensure_rgb(im)).copy()
        batch = processor(images=[image], return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.inference_mode(), _amp_context(self.device):
            e = model.get_image_features(**batch)
        e = torch.nn.functional.normalize(e.float(), dim=1).cpu().numpy()

        del model, processor, batch
        _clean_torch(self.device)
        return e

    def _predict_eva(self, image_path: Path) -> Tuple[np.ndarray, str]:
        tags, emb, names = self._extract_eva(image_path)
        cue = str(visual_cue_strings(tags, names)[0])

        selected = self.eva_head["selector"].transform(tags)
        x = np.hstack([selected, emb])
        x = self.eva_head["scaler"].transform(x)
        proba = self.eva_head["classifier"].predict_proba(x)
        aligned = _align_classifier_probabilities(
            self.eva_head["classifier"], proba, self.labels
        )[0]
        return aligned, cue

    def _predict_siglip2(self, image_path: Path) -> np.ndarray:
        emb = self._extract_siglip2(image_path)
        x = self.siglip_head["scaler"].transform(emb)
        proba = self.siglip_head["classifier"].predict_proba(x)
        aligned = _align_classifier_probabilities(
            self.siglip_head["classifier"], proba, self.labels
        )
        temperature = float(self.siglip_head.get("temperature", self.temperatures["siglip2"]))
        return softmax_np(np.log(np.clip(aligned, 1e-10, 1.0)) / temperature)[0]

    def _predict_tfidf(self, personality: str) -> np.ndarray:
        x = self.tfidf["vectorizer"].transform([personality])
        logits = _aligned_linear_scores(self.tfidf["classifier"], x, self.labels)
        temperature = float(self.tfidf.get("temperature", self.temperatures["tfidf"]))
        return softmax_np(logits / temperature)[0]

    def _load_nli_tokenizer(self):
        """Load the training tokenizer first; fall back to the saved local copy.

        The final notebook trained from the original MoritzLaurer tokenizer.
        Loading it from the original repo avoids the tokenizer-regex warning seen
        on some Transformers versions while preserving the same tokenizer source.
        """
        from transformers import AutoTokenizer

        repo = str(self.model_names["nli_large"])
        local_dir = self.models_dir / "nli_large_full_full"
        try:
            return AutoTokenizer.from_pretrained(repo)
        except Exception as remote_exc:
            warnings.warn(
                f"Could not load original NLI tokenizer from {repo}: {remote_exc}. "
                "Falling back to the tokenizer saved with the fine-tuned model.",
                RuntimeWarning,
            )
            return AutoTokenizer.from_pretrained(local_dir)

    def _predict_nli(self, name: str, personality: str, visual_cues: str) -> np.ndarray:
        import torch
        from transformers import AutoModelForSequenceClassification

        model_dir = self.models_dir / "nli_large_full_full"
        tokenizer = self._load_nli_tokenizer()
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()

        cfg = model.config
        label2id = {str(k).lower(): int(v) for k, v in cfg.label2id.items()}
        if "entailment" not in label2id or "not_entailment" not in label2id:
            raise KeyError(
                "Saved NLI model config must contain entailment and not_entailment labels; "
                f"got {cfg.label2id}"
            )
        entail_id = label2id["entailment"]
        not_id = label2id["not_entailment"]

        premise = make_premise(name, personality, visual_cues)
        template_scores: List[List[float]] = []
        for tid in range(len(HYP_TEMPLATES)):
            row: List[float] = []
            for cls in self.labels:
                hypothesis = make_hypothesis(cls, self.stable_traits[cls], tid)
                enc = tokenizer(
                    premise,
                    hypothesis,
                    truncation="only_first",
                    max_length=384,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.inference_mode():
                    logits = model(**enc).logits[0]
                row.append(float((logits[entail_id] - logits[not_id]).detach().cpu()))
            template_scores.append(row)

        temperature = float(self.temperatures["nli_large"])
        p = softmax_np(
            np.asarray(template_scores, dtype=np.float64).mean(axis=0, keepdims=True)
            / temperature
        )[0]

        del model, tokenizer
        _clean_torch(self.device)
        return p

    def _predict_modernbert(self, name: str, personality: str) -> np.ndarray:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = self.models_dir / "modernbert_full"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()

        views = [personality, f"Character: {name}. Personality: {personality}"]
        probs: List[np.ndarray] = []
        for text in views:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = model(**enc).logits.float().cpu().numpy()
            probs.append(softmax_np(logits))

        p = np.mean(probs, axis=0)
        temperature = float(self.temperatures["modernbert"])
        p = softmax_np(np.log(np.clip(p, 1e-10, 1.0)) / temperature)[0]

        del model, tokenizer
        _clean_torch(self.device)
        return p

    def predict(self, name: str, personality: str, image_path: Path | str) -> Dict[str, Any]:
        name = str(name).strip()
        personality = str(personality).strip()
        image_path = Path(image_path).expanduser().resolve()

        if not name:
            raise ValueError("character name must not be empty")
        if not personality:
            raise ValueError("personality must not be empty")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        eva_p, cues = self._predict_eva(image_path)
        siglip_p = self._predict_siglip2(image_path)
        tfidf_p = self._predict_tfidf(personality)
        nli_p = self._predict_nli(name, personality, cues)
        modern_p = self._predict_modernbert(name, personality)

        branch = {
            "nli_large": nli_p,
            "modernbert": modern_p,
            "tfidf": tfidf_p,
            "eva02": eva_p,
            "siglip2": siglip_p,
        }
        fused, decision_scores, prediction = fuse_probabilities(
            branch,
            self.core_names,
            self.weights,
            self.bias,
            self.labels,
        )

        return {
            "name": name,
            "prediction": prediction,
            # This is the same pre-bias fused probability returned by the notebook.
            "probability": {k: float(v) for k, v in zip(self.labels, fused)},
            "decision_scores": {k: float(v) for k, v in zip(self.labels, decision_scores)},
            "visual_cues": cues,
            "branch_probabilities": {
                branch_name: {label: float(v) for label, v in zip(self.labels, branch[branch_name])}
                for branch_name in self.core_names
            },
            "final_weights": {k: float(self.weights[k]) for k in self.core_names},
            "device": self.device,
            "model_version": self.config.get("version", "v3.2-final-eva-x2"),
        }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _personality_from_args(args: argparse.Namespace) -> str:
    if args.personality is not None and args.personality_file is not None:
        raise ValueError("Use either --personality or --personality-file, not both")
    if args.personality_file is not None:
        return Path(args.personality_file).read_text(encoding="utf-8").strip()
    if args.personality is not None:
        return args.personality.strip()
    raise ValueError("One of --personality or --personality-file is required")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inference for the final Dere Detector V3.2 EVA×2 multimodal ensemble."
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="Path to DereDetector_Final_Model/ (or directly to its models_v3/ folder).",
    )
    p.add_argument("--image", help="Path to the character image.")
    p.add_argument("--name", help="Character name.")
    p.add_argument("--personality", help="Personality text.")
    p.add_argument("--personality-file", help="UTF-8 text file containing the personality description.")
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device. 'auto' validates CUDA and falls back to CPU if necessary.",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the saved model bundle and exit without loading pretrained backbones.",
    )
    p.add_argument(
        "--output-json",
        help="Optional path to save the full prediction result as JSON.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        root = resolve_bundle_root(Path(args.model_dir))
        config = validate_model_bundle(root)
        if args.check_only:
            print("Model bundle: OK")
            print("Version     :", config.get("version", "v3.2-final-eva-x2"))
            print("Labels      :", ", ".join(config["labels"]))
            print("Branches    :", ", ".join(config["core_names"]))
            print("Final model : V3.2 EVA×2")
            return 0

        if args.image is None or args.name is None:
            parser.error("--image and --name are required unless --check-only is used")
        personality = _personality_from_args(args)

        predictor = DereDetectorPredictor(root, device=args.device)
        result = predictor.predict(args.name, personality, args.image)

        print(f"Prediction : {result['prediction']}")
        print("Probabilities:")
        for label, value in result["probability"].items():
            print(f"  {label:<9} {value:.6f}")
        print("Visual cues:", result["visual_cues"] or "-")
        print("Branch predictions:")
        for branch, probs in result["branch_probabilities"].items():
            pred = max(probs, key=probs.get)
            print(f"  {branch:<12} {pred:<9} ({probs[pred]:.6f})")

        if args.output_json:
            output = Path(args.output_json).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Saved JSON :", output)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
