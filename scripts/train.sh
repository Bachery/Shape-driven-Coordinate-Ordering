python trainer.py \
    \
    --just_test=False \
    --just_generate=False \
    \
    --use_cuda=True \
    --cuda=0 \
    \
    --train_size=64000 \
    --valid_size=10 \
    --batch_size=128 \
    --epoch_num=200 \
    \
    --vis_type=star \
    --reward_type=sc_sil \
    --encoder_type=rn2 \
    --data_type=dis \
    --with_label=True \
    --share_RNN=True \
    \
    --dim_num_min=16 \
    --dim_num_max=16 \
    --dim_num_step=1 \
    \
    --data_num_min=8 \
    --data_num_max=8 \
    --data_num_step=1 \
    \
    --label_num_min=2 \
    --label_num_max=2 \
    --label_num_step=1 \
    \
    --label_type=center \
    --standard=0.1 \
    \
    --note=train \
    \
    --encoder_hidden=128 \
    --decoder_hidden=128