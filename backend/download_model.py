from huggingface_hub import hf_hub_download
import os
import sys

REPO_ID  = "Sidd1312/bird-recognition-model"
SAVE_DIR = os.path.join("ml", "image")

def download_model():
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_path = os.path.join(SAVE_DIR, "bird_model.pt")
    names_path = os.path.join(SAVE_DIR, "class_names.json")

    if os.path.exists(model_path) and os.path.exists(names_path):
        print("Model already exists locally!")
        sys.stdout.flush()
        return

    token = os.environ.get("HF_TOKEN", None)

    print("Downloading model from HuggingFace...")
    sys.stdout.flush()

    hf_hub_download(
        repo_id=REPO_ID,
        filename="bird_model.pt",
        local_dir=SAVE_DIR,
        repo_type="model",
        token=token
    )
    print("Model downloaded!")
    sys.stdout.flush()

    hf_hub_download(
        repo_id=REPO_ID,
        filename="class_names.json",
        local_dir=SAVE_DIR,
        repo_type="model",
        token=token
    )
    print("Class names downloaded!")
    sys.stdout.flush()

if __name__ == "__main__":
    download_model()