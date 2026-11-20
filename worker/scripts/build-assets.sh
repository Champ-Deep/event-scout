#!/bin/sh
# Copy the static app into ./public for the Worker's asset binding. public/ is generated, never edited.
set -e
cd "$(dirname "$0")/.."
rm -rf public && mkdir -p public/.well-known
cp ../mobile-frontend.html public/index.html
cp ../card-viewer.html ../manifest.json "../Circle Scout Mascot.jpeg" public/
cp ../.well-known/assetlinks.json public/.well-known/ 2>/dev/null || true
for f in ../icon-*.png; do [ -e "$f" ] && cp "$f" public/; done
echo "assets: $(ls public | wc -l | tr -d ' ') files"
