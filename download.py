from huggingface_hub import list_repo_files, hf_hub_download
import pandas as pd

repo_id = "Baliyang/SSU_data_dash"

# See which files are available
files = list_repo_files(repo_id, repo_type="dataset")
print(files)