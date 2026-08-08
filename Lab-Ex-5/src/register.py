"""
Stage 4: register
Pushes the trained model + a model card (with the metrics that just passed
the gate) to a Hugging Face Hub model repo. Runs only from CI, only on main,
only after evaluate.py has exited 0.
"""
import os
import json
from huggingface_hub import HfApi, create_repo

REPO_ID = os.environ["HF_REPO_ID"]     
HF_TOKEN = os.environ["HF_TOKEN"]



def build_model_card(metrics: dict) -> str:
    lines = [
        "---",
        "tags: [sklearn, regression, random-forest, boston-housing, mlops-pipeline]",
        "---",
        "# Model",
        "",
        "Boston Housing price regressor trained automatically via GitHub Actions CI/CD.",
        "",
        "## Metrics",
        "",
    ]
    for model, values in metrics.items():
        lines.append(f"### {model.replace('_', ' ').title()}")
        for metric, value in values.items():
            lines.append(f"- **{metric}**: {value:.4f}")
    return "\n".join(lines)


def main():
    with open("metrics.json") as f:
        metrics = json.load(f)

    api = HfApi(token=HF_TOKEN)
    create_repo(REPO_ID, token=HF_TOKEN, exist_ok=True)

    with open("model/README.md", "w") as f:
        f.write(build_model_card(metrics))

    for path in ["model/model.joblib", "model/features.json", "data/scaler.joblib", "model/README.md"]:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=("scaler.joblib" if path == "data/scaler.joblib" else os.path.basename(path)),
            repo_id=REPO_ID,
            token=HF_TOKEN,
        )

    print(f"Model pushed to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
