#!/usr/bin/env bash
# Creates the four SEBI collections in R2R.
# Run after the stack is up: bash scripts/setup_collections.sh

R2R_URL="${R2R_URL:-http://localhost:7272}"

COLLECTIONS=("sebi_retail" "sebi_aif" "sebi_fpi" "sebi_general")

echo "Creating SEBI collections at $R2R_URL ..."

for col in "${COLLECTIONS[@]}"; do
  resp=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$R2R_URL/v3/collections" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$col\"}")
  echo "  $col → HTTP $resp"
done

echo "Done."
