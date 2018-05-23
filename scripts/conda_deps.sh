set -ex
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c pytorch magma-cuda80 # or magma-cuda90 if CUDA 9
conda install pytorch torchvision -c pytorch  # install pytorch; if you want to use cuda90, add cuda90
conda install -c conda-forge dominate  # install dominate
conda install -c conda-forge visdom    # install visdom
