import os
from huggingface_hub import hf_hub_download


def download_mpt_quant(destination_folder: str, repo_id: str, model_filename: str):
    local_path = os.path.abspath(destination_folder)
    return hf_hub_download(
        repo_id=repo_id,
        filename=model_filename,
        local_dir=local_path,
        local_dir_use_symlinks=False
    )


if __name__ == "__main__":
    """full url: https://huggingface.co/TheBloke/mpt-30B-chat-GGML/blob/main/mpt-30b-chat.ggmlv0.q4_1.bin"""

    repo_id = "TheBloke/mpt-30B-chat-GGML"
    model_filename = "mpt-30b-chat.ggmlv0.q4_1.bin"
    destination_folder = "models"
    download_mpt_quant(destination_folder, repo_id, model_filename)
