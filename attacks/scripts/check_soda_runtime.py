from __future__ import annotations

import ast
from importlib import metadata
import json
from pathlib import Path
import re
from typing import Any


EXPECTED_VERSIONS = {
    "transformers": "5.14.1",
    "trl": "1.9.2",
    "peft": "0.15.2",
    "accelerate": "1.14.0",
    "datasets": "5.0.1",
    "torch": "2.11.0+cu128",
}


def package_source(distribution: str, relative_path: str) -> Path:
    return Path(metadata.distribution(distribution).locate_file(relative_path)).resolve()


def class_node(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise RuntimeError(f"Could not find class {name} in {path}")


def class_fields(path: Path, name: str) -> set[str]:
    return {
        node.target.id
        for node in class_node(path, name).body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def method_parameters(path: Path, class_name: str, method_name: str) -> set[str] | None:
    for node in class_node(path, class_name).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            return {
                argument.arg
                for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            }
    return None


def require_parameters(
    errors: list[str],
    path: Path,
    class_name: str,
    method_name: str,
    required: set[str],
) -> None:
    parameters = method_parameters(path, class_name, method_name)
    if parameters is None:
        errors.append(f"{class_name}.{method_name} is missing")
        return
    missing = sorted(required - parameters)
    if missing:
        errors.append(f"{class_name}.{method_name} missing parameters: {missing}")


def main() -> int:
    errors: list[str] = []
    versions: dict[str, str] = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            errors.append(f"{package} is not installed")
            continue
        versions[package] = actual
        if actual != expected:
            errors.append(f"{package}=={actual} (expected {expected})")

    if errors:
        raise RuntimeError("SODA package compatibility check failed:\n- " + "\n- ".join(errors))

    paths = {
        "dpo_config": package_source("trl", "trl/trainer/dpo_config.py"),
        "dpo_trainer": package_source("trl", "trl/trainer/dpo_trainer.py"),
        "training_args": package_source("transformers", "transformers/training_args.py"),
        "trainer_callback": package_source("transformers", "transformers/trainer_callback.py"),
        "peft_model": package_source("peft", "peft/peft_model.py"),
        "lora_config": package_source("peft", "peft/tuners/lora/config.py"),
    }
    dpo_config_fields = class_fields(paths["dpo_config"], "DPOConfig")
    training_argument_fields = class_fields(paths["training_args"], "TrainingArguments")
    required_dpo_config_fields = {
        "output_dir", "beta", "learning_rate", "num_train_epochs",
        "per_device_train_batch_size", "gradient_accumulation_steps", "max_length",
        "bf16", "gradient_checkpointing", "max_grad_norm", "lr_scheduler_type",
        "warmup_steps", "save_strategy", "logging_steps",
    }
    missing_config_fields = sorted(required_dpo_config_fields - dpo_config_fields - training_argument_fields)
    if missing_config_fields:
        errors.append(f"DPOConfig missing fields: {missing_config_fields}")

    require_parameters(
        errors, paths["dpo_trainer"], "DPOTrainer", "__init__",
        {"model", "ref_model", "args", "train_dataset", "processing_class", "callbacks", "peft_config"},
    )
    require_parameters(
        errors, paths["peft_model"], "PeftModel", "from_pretrained",
        {"model", "model_id", "adapter_name", "is_trainable"},
    )
    require_parameters(
        errors, paths["peft_model"], "PeftModel", "add_adapter",
        {"adapter_name", "peft_config"},
    )
    for method_name in ("disable_adapter", "set_adapter"):
        if method_parameters(paths["peft_model"], "PeftModel", method_name) is None:
            errors.append(f"PeftModel.{method_name} is missing")
    if method_parameters(paths["trainer_callback"], "TrainerCallback", "on_pre_optimizer_step") is None:
        errors.append("TrainerCallback.on_pre_optimizer_step is missing")

    trainer_source = paths["dpo_trainer"].read_text(encoding="utf-8")
    trl_lora_fields = set(re.findall(r"default_config\.([A-Za-z_][A-Za-z0-9_]*)", trainer_source))
    peft_lora_fields = class_fields(paths["lora_config"], "LoraConfig")
    missing_lora_fields = trl_lora_fields - peft_lora_fields
    unsupported_lora_fields = sorted(missing_lora_fields - {"target_parameters"})
    if unsupported_lora_fields:
        errors.append(f"TRL reads unsupported PEFT LoraConfig fields: {unsupported_lora_fields}")

    if errors:
        raise RuntimeError("SODA API compatibility check failed:\n- " + "\n- ".join(errors))
    report: dict[str, Any] = {
        "versions": versions,
        "dpo_config_fields_checked": sorted(required_dpo_config_fields),
        "trl_lora_config_fields": sorted(trl_lora_fields),
        "missing_peft_lora_config_fields": sorted(missing_lora_fields),
        "target_parameters_compatibility": "required" if "target_parameters" in missing_lora_fields else "native",
        "imports_executed": False,
    }
    print(json.dumps({"soda_static_compatibility": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
