Code of "Partial-label Learning with Semantic Label Representations" KDD2022.

requires:
python3.9
torch1.12
python-transformers


dataset_dir:
../datasets/

parameters:
                                cifar10   cifar100 cifar100-H   cub200
beta:                              1        0.01      0.05       0.01
sigma:                            0.1       0.1       0.1        0.15

run:
python train_cifar10.py -partial_rate 0.1 -loss_weight 1 -n_map 2 -sigma 0.1 -n_class 10

python train.py -dataset cifar100 -partial_rate 0.01 -loss_weight 0.01 -n_map 2 -sigma 0.1 -n_class 100

python train.py -dataset cifar100-H -partial_rate 0.1 -loss_weight 0.05 -n_map 3 -sigma 0.1 -n_class 100

python train.py -dataset cub -partial_rate 0.01 -loss_weight 0.01 -n_map 3 -sigma 0.15 -n_class 200




