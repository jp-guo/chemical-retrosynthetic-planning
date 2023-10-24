#!/bin/bash

dropbox=../dropbox
data_name=schneider50k
tpl_name=default

save_dir="../logs/eval-gln-results"

if [ ! -e $save_dir ];
then
    mkdir "$save_dir"
fi

export CUDA_VISIBLE_DEVICES=0

python eval_gln.py \
  -dropbox $dropbox \
  -data_name $data_name \
  -save_dir $save_dir \
  -model_for_test ../checkpoints/gln.ckpt \
  -tpl_name $tpl_name \
  -f_atoms $dropbox/cooked_$data_name/atom_list.txt \
  -topk 50 \
  -beam_size 50 \
  -gpu 0 \

