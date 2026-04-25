from huggingface_hub import HfApi
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_USERNAME = "Sidd1312"
REPO_NAME   = "bird-recognition-model"
MODEL_PATH  = os.path.join("ml", "image", "bird_model.pt")
NAMES_PATH  = os.path.join("ml", "image", "class_names.json")

api    = HfApi()
REPO_ID = HF_USERNAME + "/" + REPO_NAME

print("Creating repository...")
api.create_repo(
    repo_id=REPO_NAME,
    token=HF_TOKEN,
    repo_type="model",
    exist_ok=True
)
print("Repository ready!")

print("Uploading model weights (~200MB)...")
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="bird_model.pt",
    repo_id=REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)
print("Model uploaded!")

print("Uploading class names...")
api.upload_file(
    path_or_fileobj=NAMES_PATH,
    path_in_repo="class_names.json",
    repo_id=REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)
print("Class names uploaded!")

print("Done! https://huggingface.co/" + REPO_ID)