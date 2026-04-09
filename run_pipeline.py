import os
import subprocess
import shlex
import glob
import time
import enum
from typing import Optional

class SSLStage(enum.Enum):
    PRETRAIN = "main_pretrain.py"
    LINEAR_EVAL = "adv_linear_eval.py"
    EXPORT = "export_model.py"

class SupervisedStage(enum.Enum):
    TRAIN = "adv_supervised_learning.py"
    EXPORT = "export_model.py"

class PretextVariation(enum.Enum):
    STANDARD = "adversarial_pretext=False"
    ADVERSARIAL = "adversarial_pretext=True"

class VariationsSSLStage(enum.Enum):
    #PRETEXT_ONLY = ["finetune=False", "adversarial=False"]
    #DOWNSTREAM_ADVERSARIAL_FROZEN = ["finetune=False", "adversarial=True"]
    #DOWNSTREAM_ADVERSARIAL_FINETUNE = ["finetune=True", "adversarial=True"]
    DOWNSTREAM_CLEAN_FINETUNE = ["finetune=True", "adversarial=False"]

class VariationsSupervisedStage(enum.Enum):
    STANDARD = ["finetune=True", "adversarial=False"]
    ADVERSARIAL = ["finetune=True", "adversarial=True"]

EXPERIMENTS_ROOT = "experiments"
DEFAULT_CONFIG_PATH = "scripts/linear/cifar"
DEFAULT_CONFIG_NAME = "3sat.yaml"
PRETRAIN_CONFIG_PATH = "scripts/pretrain/cifar"
PRETRAIN_CONFIG_NAME = "3sat.yaml"

def find_latest_checkpoint(pattern: str, created_after: Optional[float] = None) -> Optional[str]:
    candidates = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
    if not candidates:
        return None

    if created_after is not None:
        recent = [p for p in candidates if os.path.getmtime(p) >= created_after]
        if recent:
            candidates = recent

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

def run_command(command: str):
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )
        stdout, stderr = process.communicate()
        if stdout:
            print(stdout.decode().strip())
        if stderr:
            print(stderr.decode().strip())

        if process.returncode != 0:
            raise Exception(
                f"Command failed with return code {process.returncode}: {command}"
            )
    except Exception as e:
        print(f"An error occurred while running command: {command}\nError: {e}")
        raise Exception(e)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_export_paths(stage_dir: str) -> tuple[str, str]:
    pth_output_path = os.path.join(stage_dir, "model_full.pth")
    torchscript_output_path = os.path.join(stage_dir, "model_full.pt")
    return pth_output_path, torchscript_output_path


def quote_arg(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace(":", "\\:")
        .replace(",", "\\,")
    )
    return "'" + escaped.replace("'", "'\"'\"'") + "'"

def run_ssl_pipeline(seeds: list[int]):
    try:
        # to avoid training the backbone multiple times with the same configuration
        for pretrain_seed in seeds:
            print(f"Training backbone with seed: {pretrain_seed} / {len(seeds)}")
            pretrain_start = time.time()
            run_command(
                " ".join(
                    [
                        "python3.10",
                        SSLStage.PRETRAIN.value,
                        "--config-path",
                        PRETRAIN_CONFIG_PATH,
                        "--config-name",
                        PRETRAIN_CONFIG_NAME,
                        f"seed={pretrain_seed}",
                        PretextVariation.STANDARD.value,
                    ]
                )
            )

            pretrain_ckpt = find_latest_checkpoint(
                "trained_models/**/*.ckpt", created_after=pretrain_start
            )
            if not pretrain_ckpt:
                raise Exception("No pretrain checkpoint found after pretraining stage.")

            for variation in VariationsSSLStage:
                variation_slug = variation.name.lower()
                print(f"Running variation: {variation.name}")
                # for each fixed backbone, we run the downstream with different seeds
                for downstream_seed in seeds:
                    print(f"Running seed: {downstream_seed} / {len(seeds)}")
                    config_args = " ".join(variation.value)
                    downstream_start = time.time()
                    run_command(
                        " ".join(
                            [
                                "python3.10",
                                SSLStage.LINEAR_EVAL.value,
                                "--config-path",
                                DEFAULT_CONFIG_PATH,
                                "--config-name",
                                DEFAULT_CONFIG_NAME,
                                f"seed={downstream_seed}",
                                f"pretrained_feature_extractor={quote_arg(pretrain_ckpt)}",
                                config_args,
                            ]
                        )
                    )

                    downstream_ckpt = find_latest_checkpoint(
                        "trained_models/**/*.ckpt", created_after=downstream_start
                    )
                    if not downstream_ckpt:
                        raise Exception("No downstream checkpoint found after linear eval.")

                    export_dir = os.path.join(
                        EXPERIMENTS_ROOT,
                        f"backbone_seed{pretrain_seed}",
                        "ssl",
                        variation_slug,
                        f"seed{downstream_seed}",
                    )
                    ensure_dir(export_dir)
                    pth_output_path, torchscript_output_path = build_export_paths(export_dir)

                    run_command(
                        " ".join(
                            [
                                "python3.10",
                                SSLStage.EXPORT.value,
                                "--config-path",
                                DEFAULT_CONFIG_PATH,
                                "--config-name",
                                DEFAULT_CONFIG_NAME,
                                f"seed={downstream_seed}",
                                f"pretrained_feature_extractor={quote_arg(downstream_ckpt)}",
                                f"+export.pth_output_path={quote_arg(pth_output_path)}",
                                f"+export.torchscript_output_path={quote_arg(torchscript_output_path)}",
                            ]
                        )
                    )
    except Exception as e:
        print(f"An error occurred on run_ssl_pipeline: {e}")
        raise Exception(e)

def run_supervised_pipeline(seeds: list[int]):
    try:
        for seed in seeds:
            print(f"Running seed: {seed} / {len(seeds)}")
            train_start = time.time()
            config_args = " ".join(VariationsSupervisedStage.STANDARD.value)
            run_command(
                " ".join(
                    [
                        "python3.10",
                        SupervisedStage.TRAIN.value,
                        "--config-path",
                        DEFAULT_CONFIG_PATH,
                        "--config-name",
                        DEFAULT_CONFIG_NAME,
                        f"seed={seed}",
                        config_args,
                    ]
                )
            )

            train_ckpt = find_latest_checkpoint(
                "trained_models/**/*.ckpt", created_after=train_start
            )
            if not train_ckpt:
                raise Exception("No supervised checkpoint found after training stage.")

            export_dir = os.path.join(EXPERIMENTS_ROOT, "supervised", f"seed{seed}")
            ensure_dir(export_dir)
            pth_output_path, torchscript_output_path = build_export_paths(export_dir)

            run_command(
                " ".join(
                    [
                        "python3.10",
                        SupervisedStage.EXPORT.value,
                        "--config-path",
                        DEFAULT_CONFIG_PATH,
                        "--config-name",
                        DEFAULT_CONFIG_NAME,
                        f"seed={seed}",
                        f"pretrained_feature_extractor={quote_arg(train_ckpt)}",
                        f"+export.pth_output_path={quote_arg(pth_output_path)}",
                        f"+export.torchscript_output_path={quote_arg(torchscript_output_path)}",
                    ]
                )
            )
    except Exception as e:
        print(f"An error occurred on run_supervised_pipeline: {e}")
        raise Exception(e)

def main():
    seeds = [0,1,2] # [i for i in range(15)]
    print("Starting the pipeline...")
    print("Running self-supervised learning stage...")
    run_ssl_pipeline(seeds)

    print("Running supervised learning stage...")
    run_supervised_pipeline(seeds)

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()