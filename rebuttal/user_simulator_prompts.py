#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shim.

`prepare_doubao_contexts.py` / `evaluate_fixed_contexts.py` import
`user_simulator_prompts`, but the actual prompt definitions live in
`different_user_simulation/user_sim_prompts.py`. Re-export them here so the
long-context scripts resolve the module without duplicating the prompts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).resolve().parent / "different_user_simulation")))

from user_sim_prompts import (  # noqa: E402,F401
    available_styles,
    build_user_simulator_prompt,
    sanitize_user_response,
    strip_role_persona_prefix,
)
