from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from importlib import metadata


BASE_PACKAGES = [
    "yaml",
    "numpy",
    "tqdm",
    "datasets",
    "torch",
    "transformers",
    "accelerate",
]
TRL_PACKAGES = ["peft", "trl"]
VLLM_PACKAGES = ["vllm"]
PACKAGE_DISTS = {
    "yaml": "PyYAML",
}
TARGET_VERSIONS = {
    "transformers": "5.14.1",
    "trl": "1.9.2",
    "peft": "0.15.2",
    "accelerate": "1.14.0",
    "datasets": "5.0.1",
    "torch": "2.11.0+cu128",
    "vllm": "0.26.0",
}


def dist_name(import_name: str) -> str:
    return PACKAGE_DISTS.get(import_name, import_name)


def package_version(import_name: str) -> str:
    try:
        return metadata.version(dist_name(import_name))
    except metadata.PackageNotFoundError:
        return "unknown"


def import_package(import_name: str) -> object | None:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:
        print(f"[missing] {import_name}: {exc}", file=sys.stderr)
        return None
    version = package_version(import_name)
    target = TARGET_VERSIONS.get(import_name)
    suffix = ""
    if target is not None and version != target:
        suffix = f" (target: {target})"
    print(f"[ok] {import_name} {version}{suffix}")
    return module


def supports_parameter(callable_obj: object, parameter: str) -> bool:
    try:
        return parameter in inspect.signature(callable_obj).parameters
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the attack pipeline Python environment.")
    parser.add_argument("--require-trl", action="store_true", help="Require TRL/PEFT for QEDKS, Model Leeching, and SODA training.")
    parser.add_argument("--require-vllm", action="store_true", help="Require vLLM for endpoint-backed smoke scripts that start a local server.")
    parser.add_argument("--strict-versions", action="store_true", help="Fail when installed package versions differ from the pinned smoke-test versions.")
    args = parser.parse_args()

    required = list(BASE_PACKAGES)
    if args.require_trl:
        required.extend(TRL_PACKAGES)
    if args.require_vllm:
        required.extend(VLLM_PACKAGES)

    missing = []
    version_mismatches = []
    modules: dict[str, object] = {}
    for package in required:
        module = import_package(package)
        if module is None:
            missing.append(package)
        else:
            modules[package] = module
            target = TARGET_VERSIONS.get(package)
            if args.strict_versions and target is not None and package_version(package) != target:
                version_mismatches.append((package, package_version(package), target))

    if missing:
        print("\nInstall missing packages, for example:", file=sys.stderr)
        print("  pip install -r attacks/requirements-smoke.txt", file=sys.stderr)
        return 1

    if version_mismatches:
        print("\nVersion mismatches against attacks/requirements-smoke.txt:", file=sys.stderr)
        for package, actual, target in version_mismatches:
            print(f"  {package}: installed {actual}, target {target}", file=sys.stderr)
        print("\nInstall the pinned smoke environment, for example:", file=sys.stderr)
        print("  pip install -r attacks/requirements-smoke.txt", file=sys.stderr)
        return 1

    errors = []
    transformers = modules.get("transformers")
    if transformers is not None:
        trainer = getattr(transformers, "Trainer", None)
        if trainer is None or not supports_parameter(trainer.__init__, "processing_class"):
            errors.append("transformers.Trainer must accept processing_class")
        else:
            print("[compat] transformers.Trainer accepts required processing_class")

    trl = modules.get("trl")
    if trl is not None:
        sft_trainer = getattr(trl, "SFTTrainer", None)
        sft_config = getattr(trl, "SFTConfig", None)
        dpo_trainer = getattr(trl, "DPOTrainer", None)
        dpo_config = getattr(trl, "DPOConfig", None)
        if sft_trainer is None or not supports_parameter(sft_trainer.__init__, "processing_class"):
            errors.append("trl.SFTTrainer must accept processing_class")
        else:
            print("[compat] trl.SFTTrainer accepts required processing_class")
        if dpo_trainer is None or not supports_parameter(dpo_trainer.__init__, "processing_class"):
            errors.append("trl.DPOTrainer must accept processing_class")
        else:
            print("[compat] trl.DPOTrainer accepts required processing_class")
        if sft_config is None:
            errors.append("trl.SFTConfig must exist")
        else:
            for field in ["learning_rate", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "save_strategy", "logging_steps"]:
                if not supports_parameter(sft_config.__init__, field):
                    errors.append(f"trl.SFTConfig must accept {field}")
        if dpo_config is None:
            errors.append("trl.DPOConfig must exist")
        else:
            for field in ["max_length", "beta", "learning_rate", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "save_strategy", "logging_steps"]:
                if not supports_parameter(dpo_config.__init__, field):
                    errors.append(f"trl.DPOConfig must accept {field}")
            if supports_parameter(dpo_config.__init__, "max_prompt_length"):
                errors.append("trl.DPOConfig should not expose legacy max_prompt_length in the pinned 1.9.2 API; prompt truncation is handled before training")

    if errors:
        print("\nEnvironment API mismatch:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())