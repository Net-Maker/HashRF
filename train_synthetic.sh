LIST=("lego" "materials" "mic" "ship")
for ELEMENT in "${LIST[@]}";do
  echo "run-${ELEMENT}"
  ELEMENT_FILE="${ELEMENT}.txt"
  python train.py --config configs/"${ELEMENT_FILE}"
  python train.py --config configs/"${ELEMENT_FILE}"
  python train.py --config configs/"${ELEMENT_FILE}"
  python train.py --config configs/"${ELEMENT_FILE}"
  done