#!/usr/bin/env bash
# Build the FMC manual from its master document docs/FMC_manual.tex
#   docs/FMC_manual.pdf   pdflatex + bibtex (through latexmk)
#   README.md             pandoc + tex2gfm.lua (written at the repository root)
#
# Usage (from any directory):   bash docs/build_manual.sh [all|pdf|readme]
# Requirements: see docs/README_build.md
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
what="${1:-all}"

build_pdf() {
  latexmk -pdf -interaction=nonstopmode -halt-on-error FMC_manual.tex
  latexmk -c FMC_manual.tex > /dev/null 2>&1      # remove auxiliary files, keep the PDF
  rm -f FMC_manual.bbl
  echo "Created docs/FMC_manual.pdf"
}

build_readme() {
  version=$(sed -n 's/^\\newcommand{\\fmcversion}{\(.*\)}.*/\1/p' FMC_manual.tex)
  date=$(sed -n 's/^\\newcommand{\\fmcdate}{\(.*\)}.*/\1/p' FMC_manual.tex)
  body=$(mktemp "${TMPDIR:-/tmp}/fmc_body_XXXXXX.md")
  trap 'rm -f "$body"' EXIT
  pandoc FMC_manual.tex -f latex -t gfm --wrap=none \
         --lua-filter=tex2gfm.lua --citeproc --lua-filter=unwrap_refs.lua \
         --bibliography=FMC_manual.bib --csl=elsevier-harvard.csl \
         -o "$body"
  { sed "s/@VERSION@/${version}/; s/@DATE@/${date}/" readme_header.md; echo; cat "$body"; } > ../README.md
  echo "Created README.md (version ${version})"
}

case "$what" in
  pdf)    build_pdf ;;
  readme) build_readme ;;
  all)    build_pdf; build_readme ;;
  *)      echo "Usage: bash docs/build_manual.sh [all|pdf|readme]"; exit 1 ;;
esac
