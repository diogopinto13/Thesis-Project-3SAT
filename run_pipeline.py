import os
import subprocess
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

class VariationsStage(enum.Enum):
    PRETEXT_ONLY = ["finetune=False", "adversarial=False"]
    DOWNSTREAM_ADVERSARIAL_FROZEN = ["finetune=False", "adversarial=True"]
    DOWNSTREAM_ADVERSARIAL_FINETUNE = ["finetune=True", "adversarial=True"]

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
        stderr = process.stderr.read()
        if stderr:
            print(stderr.decode().strip())
        
        if process.returncode != 0:
            raise Exception(f"Command failed with return code {process.returncode}: {command}")
    except Exception as e:
        print(f"An error occurred while running command: {command}\nError: {e}")
        raise Exception(e)

def run_ssl_pipeline(seeds: list[int]):
    try:
        #to avoid training the backbone multiple times with the same configuration since we only change the downstream
        for seed in seeds:
            print(f"Training backbone with seed: {seed} / {len(seeds)}")
            run_command(f"python3.10 {SSLStage.PRETRAIN.value} --config-path --config-path scripts/pretrain/cifar --config-name 3sat.yaml --seed={seed}")
            for variation in VariationsStage:
                print(f"Running variation: {variation.name}")
                #for each fixed backbone, we run the downstream with different settings for different seeds
                for seed in seeds:
                    print(f"Running seed: {seed} / {len(seeds)}")
                    #override configs for the downstream stage
                    config_args = " ".join([f"--{arg}" for arg in variation.value])
                    run_command(f"python3.10 {SSLStage.LINEAR_EVAL.value} --config-path --config-path scripts/linear/cifar --config-name 3sat.yaml --seed={seed} {config_args}")
                    config_args += (f" --save_name={variation.name}_seed_{seed}" + f"--checkpoint_path=checkpoints/pretrain/cifar/3sat/")
                    run_command(f"python3.10 {SSLStage.EXPORT.value} --config-path --config-path scripts/export/cifar --config-name 3sat.yaml --seed={seed} {config_args}")
    except Exception as e:
        print(f"An error occurred on run_ssl_pipeline: {e}")
        raise Exception(e)

def run_supervised_pipeline(seeds: list[int]):
    try:
        for stage in SupervisedStage:
            print(f"Running stage: {stage.name}")
            for seed in seeds:
                print(f"Running seed: {seed} / {len(seeds)}")
                config_args = f"--seed={seed} --save_name={stage.name}_seed_{seed}"
                command = f"python3.10 {stage.value} --config-path --config-path scripts/linear/cifar --config-name 3sat.yaml {config_args}"
                run_command(command)
    except Exception as e:
        print(f"An error occurred on run_supervised_pipeline: {e}")
        raise Exception(e)

def main():
    seeds = [0, 1, 2, 3, 4]
    print("Starting the pipeline...")
    print("Running self-supervised learning stage...")
    run_ssl_pipeline(seeds)

    print("Running supervised learning stage...")
    run_supervised_pipeline(seeds)

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()