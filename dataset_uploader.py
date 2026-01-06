from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your dataset files
upload_folder(folder_path=".", repo_id="victoroseji/fnol_claims_dataset", repo_type="dataset")
