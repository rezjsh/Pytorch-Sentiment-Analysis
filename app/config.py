from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppPaths:
    checkpoints: Path
    reports: Path


def ensure_paths() -> AppPaths:
    ckpt = Path("checkpoints")
    rep = Path("reports")
    ckpt.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    return AppPaths(checkpoints=ckpt, reports=rep)
