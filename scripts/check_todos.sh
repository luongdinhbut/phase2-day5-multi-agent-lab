#!/usr/bin/env bash
set -euo pipefail
grep -R "IMPLEMENT_ME\|REPLACE_ME" -n src tests docs || true
