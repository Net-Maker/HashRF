LIST=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
for ELEMENT in "${LIST[@]}";do
  echo "run-${ELEMENT}"
  ELEMENT_FILE="${ELEMENT}-cp.txt"
  python train.py --config configs/"${ELEMENT_FILE}"
  done