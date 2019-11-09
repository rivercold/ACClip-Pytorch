export GLUE_DIR=./glue_data

python run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --optimizer SGD \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/ \
  --overwrite_output_dir
