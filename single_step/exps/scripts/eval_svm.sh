python train_eval_svm.py \
  --data_dir ../dropbox/fingerprint \
  --save_dir ../logs/eval-svm-results \
  --kernel linear \
  --C 0.00001 \
  --do_eval \
  --pretrained ../checkpoints/svm.model