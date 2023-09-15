##### This script will be executed after the container-building process. #####

pip install -r ~/Lab6/requirements.txt

git clone git@github.com:YuZhong-Chen/.setup_env.git ~/.setup_env
cd ~/.setup_env && ./install.sh