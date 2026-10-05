"""Exhaustiveness helper for enum switches. Python 3.10 has no typing.Never."""

from __future__ import annotations

from typing import NoReturn

try:
    from typing import Never
except ImportError:  # Python 3.10

    class Never:
        """Stand-in so callers import on Python 3.10. Uninhabited at runtime."""

        def __new__(cls, *_args: object, **_kwargs: object) -> Never:
            raise TypeError("Never cannot be instantiated")


def assert_never(value: Never) -> NoReturn:
    raise RuntimeError(f"Unhandled variant: {value!r}")
