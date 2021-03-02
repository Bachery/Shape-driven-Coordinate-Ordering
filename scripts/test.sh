python trainer.py \
    \
    --just_test=True \
    --just_generate=False \
    \
    --use_cuda=True \
    --cuda=0 \
    \
    --train_size=10 \
    --valid_size=1000 \
    --batch_size=128 \
    --epoch_num=1 \
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
    --note=valid-prob \
    \
    --encoder_hidden=128 \
    --decoder_hidden=128 \
    --checkpoint=./pretrained/star/16/sc_sil-rn2-dis-True-True-[16-16]d-[8-8]n-[2-2]c-center-0.1-note-train/checkpoints/199
