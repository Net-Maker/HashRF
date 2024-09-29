# LIST=( "spaceship" "streamtrain"  "wineholder" "robot")
LIST=( "lifestyle" "palace" "toad" "chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
for ELEMENT in "${LIST[@]}";do
  echo "run-${ELEMENT}"
  ELEMENT_FILE="${ELEMENT}.txt"
  CUDA_MPS_PIPE_DIRECTORY="" CUDA_MPS_LOG_DIRECTORY="" CUDA_VISIBLE_DEVICES=0 python train.py --config configs/"${ELEMENT_FILE}"
  done