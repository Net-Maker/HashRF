LIST=(  "toad" "palace" "lifestyle" "bike")
# "spaceship" "streamtrain"  "wineholder" "robot"
for ELEMENT in "${LIST[@]}";do
  echo "run-${ELEMENT}"
  ELEMENT_FILE="${ELEMENT}.txt"
  python train.py --config configs/"${ELEMENT_FILE}"
  done