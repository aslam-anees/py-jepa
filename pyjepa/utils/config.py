"""Tiny YAML config helper. We deliberately avoid Hydra/OmegaConf so the lib stays
dependency-light; if the user wants those, they can build configs externally and
pass dicts in. ``Config`` is just a dot-accessible dict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import yaml


class Config(dict):
    """A dict that also supports ``cfg.foo.bar`` access and nested merging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            self[k] = _wrap(v)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = _wrap(value)

    def __delattr__(self, key: str) -> None:
        del self[key]

    def merge(self, other: dict) -> "Config":
        """Deep-merge ``other`` into self (mutating). Returns self for chaining."""
        for k, v in other.items():
            if isinstance(v, dict) and isinstance(self.get(k), dict):
                if not isinstance(self[k], Config):
                    self[k] = Config(self[k])
                self[k].merge(v)
            else:
                self[k] = _wrap(v)
        return self

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        for k, v in self.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        # Support dotted lookup: cfg.get("data.batch_size")
        if "." in key:
            cur: Any = self
            for part in key.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur
        return super().get(key, default)


def _wrap(v: Any) -> Any:
    if isinstance(v, dict) and not isinstance(v, Config):
        return Config(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def load_config(path: Union[str, Path], overrides: Optional[dict] = None) -> Config:
    """Load YAML/JSON config from disk, then apply optional override dict."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".json",):
        data = json.loads(text)
    else:
        data = yaml.safe_load(text) or {}
    cfg = Config(data)
    if overrides:
        cfg.merge(overrides)
    return cfg


def save_config(cfg: Union[Config, dict], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    if path.suffix == ".json":
        path.write_text(json.dumps(data, indent=2))
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False))
