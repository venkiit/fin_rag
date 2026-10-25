import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
repo_id = sys.argv[1]                                   # e.g. venkiit/fin-rag
api = HfApi(token=os.environ["HF_TOKEN"])

api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=[".venv/*", ".env", ".git/*", "data/models/*", "data/uploads/*",
                     "data/indexes/*", "eval/*", "**/__pycache__/*", "deploy_hf.py"],
    commit_message="Deploy from local project",
)
api.add_space_secret(repo_id, "OPENROUTER_API_KEY", os.environ["OPENROUTER_API_KEY"])
api.add_space_variable(repo_id, "LLM_PROVIDER", os.getenv("LLM_PROVIDER", "openrouter"))
api.add_space_variable(repo_id, "LLM_MODEL", os.getenv("LLM_MODEL", "google/gemini-2.5-flash"))

user, name = repo_id.split("/")
print(f"\nUploaded. Build logs:  https://huggingface.co/spaces/{repo_id}")
print(f"App URL (after build): https://{user}-{name}.hf.space".lower())