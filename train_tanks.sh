LIST=("barn" "caterpillar" "family" "ignatius" "truck")
for ELEMENT in "${LIST[@]}";do
  echo "run-${ELEMENT}"
  ELEMENT_FILE="${ELEMENT}.txt"
  python train.py --config configs/"${ELEMENT_FILE}"
  done

bash train_llff.sh >> ner_version_HashRF_llff.out