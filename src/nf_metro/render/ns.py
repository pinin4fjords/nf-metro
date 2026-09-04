"""SVG class-namespace utilities shared across render modules."""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
from collections.abc import Generator

__all__ = ["ns", "class_prefix_context", "adaptive_logo_mask_ids"]

_render_class_prefix: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_render_class_prefix", default=""
)


def ns(cls: str) -> str:
    """Apply the active render namespace prefix to an SVG class name."""
    p = _render_class_prefix.get()
    return f"{p}-{cls}" if p else cls


def adaptive_logo_mask_ids(key_path: str) -> tuple[str, str]:
    """Return stable, unique SVG mask IDs for an adaptive logo pair.

    IDs are derived from the final segment of *key_path* (dark logo if
    available, else light) so the same asset always maps to the same IDs,
    avoiding collisions when multiple SVGs are inlined on one page. Only the
    final segment is keyed because *key_path* has been resolved against the
    map's source directory and so carries the absolute location of the
    checkout, which must not reach the rendered bytes. Two distinct assets
    that share a final segment share a mask harmlessly: both masks are plain
    ``light-dark()`` rects whose content does not depend on the asset.
    Returns ``(dark_mask_id, light_mask_id)``.
    """
    h = hashlib.md5(key_path.rsplit("/", 1)[-1].encode()).hexdigest()[:8]
    return ns(f"nfm-logo-mask-dark-{h}"), ns(f"nfm-logo-mask-light-{h}")


@contextlib.contextmanager
def class_prefix_context(prefix: str) -> Generator[None, None, None]:
    """Context manager that sets the SVG class namespace prefix for the duration."""
    token = _render_class_prefix.set(prefix)
    try:
        yield
    finally:
        _render_class_prefix.reset(token)
