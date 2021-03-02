python trainer.py \
    \
    --just_test=False \
    --just_generate=True \
    \
    --use_cuda=True \
    --cuda=0 \
    \
    --train_size=10 \
    --valid_size=10 \
    --batch_size=128 \
    --epoch_num=1 \
    \
    --vis_type=star \
    --reward_type=sc_sil \
    --data_type=dis \
    --with_label=True \
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
    --standard=0.1 \
    \
    --encoder_type=rn2 \
    --encoder_hidden=256 \
    --decoder_hidden=256 \
    --share_RNN=True \
    --label_type=center \
    \
    --note=generate
