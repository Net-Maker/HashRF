LIST=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
for ELEMENT in "${LIST[@]}";do
  echo "run-${ELEMENT}"
  ELEMENT_FILE="${ELEMENT}.txt"
  echo "1"
  python train.py --config configs/"${ELEMENT_FILE}"
  echo "2"
  python train.py --config configs/"${ELEMENT_FILE}"
  echo "3"
  python train.py --config configs/"${ELEMENT_FILE}"
  echo "4"
  python train.py --config configs/"${ELEMENT_FILE}"
  done