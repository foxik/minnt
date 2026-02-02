#!/bin/sh

[ $# -ge 1 ] || { echo "Usage: $0 new_version" >&2; exit 1; }
version="$1"
major="${version%%.*}"
minor_and_rest="${version#*.}"
minor="${minor_and_rest%%.*}"

BASE="$(dirname "$(dirname "$(readlink -f "$0")")")"

sed 's/version = "[^"]*"/version = "'"$version"'"/' -i $BASE/pyproject.toml
sed 's/__version__ = "[^"]*"/__version__ = "'"$version"'"/' -i $BASE/minnt/version.py
sed 's/^# Minnt [^ ]*$/# Minnt '"$version"'/' -i $BASE/docs/src/index.md
sed 's/"name": "Minnt-[^"]*"/"name": "Minnt-'"$major"."$minor"'"/' -i $BASE/docs/dashing/dashing.json
