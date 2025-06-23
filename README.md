# Code-for-KAC-master

Hardware requirement：RTX3090 (CUDA VERSION: 11.3)


# Environmental configuration：

1. Creat python 3.7.16
conda create -n dgl python=3.7.16

2. Activate
conda activate dgl

3. Install pytorch 1.12.1 cuda 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

4. Install dgl 1.1.1 cu113
pip install dgl==1.1.1+cu113 -f https://data.dgl.ai/wheels/cu113/repo.html

5. Others
pip install pandas matplotlib scikit-learn munkres

# Code running tips

Open run.py to run KAC. You can replace the dataset in the experiment as you want, and we use IMDB as an example. Notice that the data are packed in the datazip from releases.

References: "Knowledge-Aware Clustering", under review.

Written by: Mansheng Chen, chenmsh27@mail2.sysu.edu.cn 2025-06-22
