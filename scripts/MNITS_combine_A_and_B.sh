set -ex
python datasets/combine_A_and_B.py --fold_A datasets/mnist_4channel/\A --fold_B datasets/mnist_4channel/\B --fold_AB datasets/mnist_4channel/\AB
