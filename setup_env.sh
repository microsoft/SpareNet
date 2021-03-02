sudo apt-get update
sudo apt-get --yes install vim
sudo apt-get --yes install tmux

pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl --user 

PROJ_HOME=`pwd`

cd $PROJ_HOME/cuda/emd/
rm -rf build/*
python setup.py install --user

cd $PROJ_HOME/cuda/expansion_penalty/
rm -rf build/*
python setup.py install --user

cd $PROJ_HOME/cuda/MDS/
rm -rf build/*
python setup.py install --user

cd $PROJ_HOME/cuda/cubic_feature_sampling/
rm -rf build/*
python setup.py install --user

cd $PROJ_HOME/cuda/gridding/
rm -rf build/*
python setup.py install --user

cd $PROJ_HOME/cuda/gridding_loss/
rm -rf build/*
python setup.py install --user
cd $PROJ_HOME
