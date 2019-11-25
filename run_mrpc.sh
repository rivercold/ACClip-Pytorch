export GLUE_DIR=./glue_data

python run_mrpc.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --optimizer acclip \
  --do_plot \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --evaluate_during_training \
  --overwrite_output_dir \
  --output_dir ./outputs
