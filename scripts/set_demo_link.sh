#!/bin/bash
# Insert the demo-video link into README.md and both PDFs, then rebuild the PDFs.
#   bash scripts/set_demo_link.sh https://drive.example.com/your-video-link
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${1:?usage: set_demo_link.sh <url>}"
source scripts/env.sh
for f in README.md docs/templates/README.md docs/WRITEUP.md docs/templates/WRITEUP.md docs/SUMMARY.md; do
  sed -i "s#_link to be added_#<${URL}>#g" "$f"
done
$PY docs/build_pdf.py WRITEUP
$PY docs/build_pdf.py SUMMARY
grep -n "Demo video" README.md docs/WRITEUP.md docs/SUMMARY.md
