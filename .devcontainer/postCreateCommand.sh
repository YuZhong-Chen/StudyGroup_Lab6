##### This script will be executed after the container-building process. #####

pip install -r ~/Lab6/requirements.txt

# Disable showwing (bash) in terminal
conda config --set changeps1 False

git clone git@github.com:YuZhong-Chen/.setup_env.git ~/.setup_env
cd ~/.setup_env && ./install.sh