# Test the trained network on dinmension 16~24, then draw a curve.
# This script will generate a image as 'shape_task/compare/curve/curve_16n.png'
# In our paper the valid_size=10,000, but the data generation is time-consuming.
# In this example code we set the testing size as 100, 
# so the final curve maybe slightly different from our curve shown in paper.


mkdir -p data/rand
mkdir compare/curve

# run network to get test result
python run_curve.py

cd compare
python draw_curve.py
