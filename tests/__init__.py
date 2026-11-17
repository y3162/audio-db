"""Repository tests.

NeMo installs a regular top-level ``tests`` package. Without this file the
repository tree is only a namespace, and Python keeps the regular package it
finds later in ``site-packages``. This module makes the repository win when
its root is earlier on ``sys.path``. The site-packages copy stays installed.
"""
