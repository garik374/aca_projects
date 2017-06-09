./fasttext supervised -input ft_train -output model_ft -dim 50 -minCount 2 -maxn 4 -ws 3 -wordNgrams 2

./fasttext test model_ft.bin ft_test
