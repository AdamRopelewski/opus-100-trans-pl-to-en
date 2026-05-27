from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class LanguageIdRuntime:
    enabled: bool
    reason: str
    detect: Callable[[str], str | None]


def build_language_id_runtime(enabled: bool, strict_dependency: bool = False) -> LanguageIdRuntime:
    if not enabled:
        return LanguageIdRuntime(enabled=False, reason="disabled_by_config", detect=lambda _: None)

    try:
        import langid  # type: ignore
    except Exception as exc:  # pragma: no cover
        if strict_dependency:
            raise RuntimeError(
                "Language ID requested but 'langid' package missing. Install it or disable language_id filter/check."
            ) from exc
        return LanguageIdRuntime(enabled=False, reason="langid_not_installed", detect=lambda _: None)

    def _detect(text: str) -> str | None:
        if not text:
            return None
        code, _score = langid.classify(text)
        return str(code)

    return LanguageIdRuntime(enabled=True, reason="enabled", detect=_detect)
