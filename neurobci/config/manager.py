"""Load / save / list configuration profiles.

A *profile* is a single JSON file under the configs directory. The
manager is deliberately small and dependency-free (stdlib ``json``); it
performs no code execution, so a malicious config can at worst supply bad
values -- which downstream validation is expected to catch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from neurobci.config.schema import AppConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path("configs")
_SUFFIX = ".json"


class ConfigError(Exception):
    """Raised when a configuration file cannot be parsed."""


class ConfigManager:
    """Manages a directory of JSON configuration profiles."""

    def __init__(self, config_dir: Path | str = DEFAULT_CONFIG_DIR) -> None:
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    # ----- paths --------------------------------------------------------- #

    def _path_for(self, profile_name: str) -> Path:
        safe = "".join(c for c in profile_name if c.isalnum() or c in "-_")
        if not safe:
            raise ConfigError(f"Invalid profile name: {profile_name!r}")
        return self.config_dir / f"{safe}{_SUFFIX}"

    # ----- listing ------------------------------------------------------- #

    def list_profiles(self) -> list[str]:
        return sorted(p.stem for p in self.config_dir.glob(f"*{_SUFFIX}"))

    def exists(self, profile_name: str) -> bool:
        return self._path_for(profile_name).exists()

    # ----- load / save --------------------------------------------------- #

    def load(self, profile_name: str = "default") -> AppConfig:
        """Load a profile, or return defaults (and create it) if missing."""

        path = self._path_for(profile_name)
        if not path.exists():
            logger.info("Profile %r not found; creating defaults.", profile_name)
            cfg = AppConfig(profile_name=profile_name)
            self.save(cfg)
            return cfg
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Could not read {path}: {exc}") from exc
        cfg = AppConfig.from_dict(raw)
        cfg.profile_name = profile_name
        logger.info("Loaded configuration profile %r.", profile_name)
        return cfg

    def save(self, config: AppConfig) -> Path:
        path = self._path_for(config.profile_name)
        # Atomic-ish write: temp file then replace, so a crash mid-write
        # cannot corrupt an existing profile.
        tmp = path.with_suffix(_SUFFIX + ".tmp")
        tmp.write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=False),
            encoding="utf-8",
        )
        tmp.replace(path)
        logger.info("Saved configuration profile to %s.", path)
        return path

    def duplicate(self, src: str, dst: str) -> AppConfig:
        cfg = self.load(src)
        cfg.profile_name = dst
        self.save(cfg)
        return cfg

    def delete(self, profile_name: str) -> None:
        path = self._path_for(profile_name)
        if path.exists():
            path.unlink()
            logger.info("Deleted profile %r.", profile_name)
