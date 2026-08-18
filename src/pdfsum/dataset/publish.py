"""Pushes the local dataset to a private Hugging Face dataset repo, so the
Kaggle/Colab training notebook (which has no access to this local machine)
can pull it. Private by default: the report/contract source text's copyright
status isn't clearly ours to redistribute publicly (see sources.py)."""
import os
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO_ID = "makremlupin/pdf-summarizer-agent-dataset"


class MissingHfTokenError(RuntimeError):
    pass


def _api() -> HfApi:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise MissingHfTokenError(
            "HF_TOKEN is not set. MANUAL_ACTION_REQUIRED: create a free Hugging "
            "Face account, generate an access token at "
            "https://huggingface.co/settings/tokens, and export it as HF_TOKEN."
        )
    return HfApi(token=token)


def ensure_repo(repo_id: str = DEFAULT_REPO_ID, private: bool = True) -> str:
    return _api().create_repo(
        repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
    )


def upload_file(
    local_path: Path, path_in_repo: str, repo_id: str = DEFAULT_REPO_ID
) -> str:
    ensure_repo(repo_id)
    return _api().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
