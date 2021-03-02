python shape_valid.py \
    \
    --just_test=True \
    --just_generate=False \
    \
    --use_cuda=True \
    --cuda=0 \
    \
    --train_size=10 \
    --valid_size=10000 \
    --batch_size=128 \
    --epoch_num=1 \
    \
    --dim_num_min=16 \
    --dim_num_max=24 \
    --dim_num_step=1 \
    --sample_num=80 \
    --standard=0.1 \
    \
    --net_type=fcn2 \
    --hidden_size=128 \
    \
    --note=test \
    --checkpoint=./shape/24d-fcn2/checkpoints/99