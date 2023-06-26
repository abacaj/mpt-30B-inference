import os
from huggingface_hub import hf_hub_download


def download_mpt_quant(destination_folder):
    local_path = os.path.relpath(destination_folder)
    return hf_hub_download(
        repo_id="TheBloke/mpt-30B-chat-GGML",
        filename="mpt-30b-chat.ggmlv0.q4_1.bin",
        cache_dir=local_path,
    )


if __name__ == "__main__":
    """full url: https://huggingface.co/TheBloke/mpt-30B-chat-GGML/blob/main/mpt-30b-chat.ggmlv0.q4_1.bin"""

    destination_folder = "models"
    download_mpt_quant(destination_folder)
