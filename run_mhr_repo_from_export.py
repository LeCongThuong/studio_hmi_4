#!/usr/bin/env python3
"""Compatibility wrapper for official-MHR forward replay."""

from studio_hmi_4.export.official_mhr import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
