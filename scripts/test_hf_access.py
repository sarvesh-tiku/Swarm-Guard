"""Verify Hugging Face access to the gated AI Village dataset.

Usage: python scripts/test_hf_access.py
"""
from __future__ import annotations

import sys

from swarmguard.data.hf_access import REPO, HFAccessError, explain_error, token_kwarg, whoami


def main() -> int:
    try:
        user = whoami()
    except HFAccessError as e:
        print(f"FAILED: {e}")
        return 1
    print(f"Authenticated Hugging Face access: OK (user: {user})")
    print(f"Repository: {REPO}")

    from datasets import get_dataset_config_names, load_dataset

    try:
        configs = get_dataset_config_names(REPO, **token_kwarg())
        print(f"Available configs: {len(configs)}")
        test_config = "agent_goals" if "agent_goals" in configs else configs[0]
        print(f"Test config: {test_config}")
        ds = load_dataset(REPO, test_config, split="train", streaming=True, **token_kwarg())
        row = next(iter(ds))
        print(f"Test row loaded successfully ({len(row)} columns: {', '.join(list(row)[:8])}...)")
    except Exception as e:
        print(f"FAILED: {explain_error(e)}")
        print("Fix: run `hf auth login` with the account that was granted access, then retry.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
