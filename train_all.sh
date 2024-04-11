# LIST=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
# LIST=("drums" "drums" "drums" "drums" "ship" "ship" "ship" "ship")
# for ELEMENT in "${LIST[@]}";do
#   echo "run-${ELEMENT}"
#   ELEMENT_FILE="${ELEMENT}.txt"
#   python train.py --config configs/"${ELEMENT_FILE}"
#   done
# LIST=( "spaceship" "streamtrain"  "wineholder" "robot")
LIST=("bike" "lifestyle" "palace" "toad")
for ELEMENT in "${LIST[@]}";do
  echo "run-${ELEMENT}"
  ELEMENT_FILE="${ELEMENT}.txt"
  python train.py --config configs/"${ELEMENT_FILE}"
  done
# LIST=("barn" "caterpillar" "family" "ignatius" "truck")
LIST=("barn" "ignatius")
for ELEMENT in "${LIST[@]}";do
  echo "run-${ELEMENT}"
  ELEMENT_FILE="${ELEMENT}.txt"
  python train.py --config configs/"${ELEMENT_FILE}"
  done
# LIST=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
# for ELEMENT in "${LIST[@]}";do
#   echo "run-${ELEMENT}"
#   ELEMENT_FILE="${ELEMENT}.txt"
#   python train.py --config configs/"${ELEMENT_FILE}"
#   done