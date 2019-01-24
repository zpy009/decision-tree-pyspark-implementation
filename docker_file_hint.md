
# Base
apt-get update -y
apt-get install vim -y

wget https://repo.anaconda.com/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh]

bash Miniconda3-4.2.12-Linux-x86_64.sh # Many interactive steps

source /root/.bashrc

conda install pandas

conda install numpy


cp spark_env.sh /spark/conf/spark_env.sh

# Master

cp testdata.pkl

cp 5003