# LIST=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
LIST=("hotdog" "lego" "materials" "mic" "ship")
for ELEMENT in "${LIST[@]}";do
  echo "run-${ELEMENT}"
  ELEMENT_FILE="${ELEMENT}.txt"
  echo "1"
  CUDA_MPS_PIPE_DIRECTORY="" CUDA_MPS_LOG_DIRECTORY="" CUDA_VISIBLE_DEVICES=0 python train.py --config configs/"${ELEMENT_FILE}"
  # echo "2"
  # python train.py --config configs/"${ELEMENT_FILE}"
  # echo "3"
  # python train.py --config configs/"${ELEMENT_FILE}"
  # echo "4"
  # python train.py --config configs/"${ELEMENT_FILE}"
  done