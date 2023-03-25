# adult
rm -rf adult
mkdir adult
wget -P adult https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget -P adult https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
tail -n +2 adult/adult.test > tmp && mv tmp adult/adult.test && rm -rf tmp
wget -P adult https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names

# bank
rm -rf bank
mkdir bank
wget -P bank https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
unzip bank/bank.zip -d bank

# compass
rm -rf compass
mkdir compass
kaggle datasets download -d danofer/compass -p compass
unzip compass/compass.zip -d compass

# kdd
rm -rf kdd
mkdir kdd
wget -P kdd http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz
tar -xvf kdd/census.tar.gz -C kdd
