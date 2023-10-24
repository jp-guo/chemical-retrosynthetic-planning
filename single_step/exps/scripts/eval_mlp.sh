export CUDA_VISIBLE_DEVICES=0

python train_eval_mlp.py \
  --data_dir ../dropbox/fingerprint \
  --dim 512 \
  --batchSz 256 \
  --testBatchSz 512 \
  --nEpoch 40 \
  --lr 1e-4 \
  --save_dir ../logs/eval-mlp-results \
  --do_eval \
  --pretrained ../checkpoints/mlp.pth