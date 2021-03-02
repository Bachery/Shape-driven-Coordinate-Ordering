'''
Test the trained network on dinmension 16~24
In our paper the valid_size=10,000, but the data generation is time-consuming.

In this example code we set the testing size as 100, 
so the final curve maybe a little different from our curve shown on paper.
'''

import os

star_dim = 16
end_dim = 32

valid_size = 100

print(star_dim, end_dim)

# ORIG net_type=fc
for dim_num in range(star_dim, end_dim+1):
    run_str = 'python shape_valid.py --just_test=True --just_generate=False  \
            --valid_size=%d --train_size=10 --epoch_num=1 \
            --use_cuda=True \
            --cuda=0 \
            --dim_num_min=%d --dim_num_max=%d \
            --dim_num_step=1 \
            --sample_num=80 \
            --standard=0.1 \
            --net_type=fc \
            --batch_size=128 \
            --hidden_size=128 \
            --note=valid-curve \
            --checkpoint=./shape/24d-fc/checkpoints/99' % (valid_size, dim_num, dim_num)
    print(dim_num)
    os.system(run_str)

# SAMPLE net_type=fcn2
for dim_num in range(star_dim, end_dim+1):
    run_str = 'python shape_valid.py --just_test=True --just_generate=False  \
            --valid_size=%d --train_size=10 --epoch_num=1 \
            --use_cuda=True \
            --cuda=0 \
            --dim_num_min=%d --dim_num_max=%d \
            --dim_num_step=1 \
            --sample_num=80 \
            --standard=0.1 \
            --net_type=fcn2 \
            --batch_size=128 \
            --hidden_size=128 \
            --note=valid-curve \
            --checkpoint=./shape/24d-fcn2/checkpoints/99' % (valid_size, dim_num, dim_num)
    print(dim_num)
    os.system(run_str)