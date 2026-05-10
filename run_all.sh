#!/bin/bash
set -e
PY=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
mkdir -p logs
for fusion in early_fusion late_fusion; do
  for opt in Adam RMSprop AdamW; do
    log="logs/${fusion}_${opt}.log"
    if [ -f "${fusion}_${opt}_report/test_summary.json" ]; then
      echo "[skip] ${fusion} ${opt} already complete"
      continue
    fi
    echo "[run]  ${fusion} ${opt}  ->  $log"
    "$PY" -u hw6_q3_template.py run "$fusion" "$opt" > "$log" 2>&1
    echo "[done] ${fusion} ${opt}  exit=$?"
  done
done
echo "[aggregate]"
"$PY" -u hw6_q3_template.py aggregate > logs/aggregate.log 2>&1
echo "[all done]"
