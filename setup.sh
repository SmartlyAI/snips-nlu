# Install NFS protcol for snips-data
sudo apt-get install nfs-kernel-server -y
sudo chown -R $USER:$USER /snips-data
echo "/snips-data    ${DISTANT_SERVER_PRIVATE_IP}(rw,sync,no_subtree_check)" | sudo tee /etc/exports
sudo systemctl restart nfs-kernel-server

# Swarm alreay cerated with docker swarm init on azure
docker swarm init

# Create registry which will contains all images for smartly services
docker service create --name registry-smartly-swarm --publish 5000:5000 --constraint 'node.role == manager' --detach=false registry:2

# Create volumes
docker volume create --name=rabbitmq

# Cloning hestia only if it does not exist
if [ ! -d "hestia" ]; then
  git clone git@github.com:SmartlyAI/hestia.git
  cd hestia && git checkout snips && cd ..
else
  echo 'Updating hestia repository'
  cd hestia &&  git fetch && git checkout snips && git pull origin snips && cd ..
fi

# Cloning snips-nlu-parse 1.05 only if it does not exist
if [ ! -d "snips-nlu-parse-105" ]; then
  git clone git@github.com:SmartlyAI/snips-nlu-parse.git snips-nlu-parse-105
  cd snips-nlu-parse-105 && git checkout snips/1.05 && cd ..
else
  echo 'Updating snips-nlu-parse-105 repository'
  cd snips-nlu-parse-105 &&  git fetch && git checkout snips/1.05 && git pull origin snips/1.05 && cd ..
fi

# Cloning snips-nlu-train 1.05 only if it does not exist
if [ ! -d "snips-nlu-train-105" ]; then
  git clone git@github.com:SmartlyAI/snips-nlu-train.git snips-nlu-train-105
  cd snips-nlu-train-105 && git checkout snips/1.05 && cd ..
else
  echo 'Updating snips-nlu-train-105 repository'
  cd snips-nlu-train-105 &&  git fetch && git checkout snips/1.05 && git pull origin snips/1.05 && cd ..
fi

# Cloning snips-nlu-parse 1.04 only if it does not exist
if [ ! -d "snips-nlu-parse-104" ]; then
  git clone git@github.com:SmartlyAI/snips-nlu-parse.git snips-nlu-parse-104
  cd snips-nlu-parse-104 && git checkout snips/1.04 && cd ..
else
  echo 'Updating snips-nlu-parse-104 repository'
  cd snips-nlu-parse-104 &&  git fetch && git checkout snips/1.04 && git pull origin snips/1.04 && cd ..
fi

# Cloning snips-nlu-train 1.04 only if it does not exist
if [ ! -d "snips-nlu-train-104" ]; then
 git clone git@github.com:SmartlyAI/snips-nlu-train.git snips-nlu-train-104
 cd snips-nlu-train-104 && git checkout snips/1.04 && cd ..
else
  echo 'Updating snips-nlu-train-104 repository'
  cd snips-nlu-train-104 &&  git fetch && git checkout snips/1.04 && git pull origin snips/1.04 && cd ..
fi

# Cloning snips-nlu-parse 1.06 only if it does not exist
if [ ! -d "snips-nlu-parse-106" ]; then
  git clone git@github.com:SmartlyAI/snips-nlu-parse.git snips-nlu-parse-106
  cd snips-nlu-parse-106 && git checkout snips/1.06 && cd ..
else
  echo 'Updating snips-nlu-parse-106 repository'
  cd snips-nlu-parse-106 &&  git fetch && git checkout snips/1.06 && git pull origin snips/1.06 && cd ..
fi

# Cloning snips-nlu-train 1.06 only if it does not exist
if [ ! -d "snips-nlu-train-106" ]; then
 git clone git@github.com:SmartlyAI/snips-nlu-train.git snips-nlu-train-106
 cd snips-nlu-train-106 && git checkout snips/1.06 && cd ..
else
  echo 'Updating snips-nlu-train-106 repository'
  cd snips-nlu-train-106 &&  git fetch && git checkout snips/1.06 && git pull origin snips/1.06 && cd ..
fi

# Build stack
privatekey=$(cat ~/.ssh/id_rsa)
publickey=$(cat ~/.ssh/id_rsa.pub)
SSH_PRIVATE_KEY="$privatekey" SSH_PUBLIC_KEY="$publickey" docker-compose build

# Push and launch snips_nlu stack
docker-compose push && env $(cat .env | grep ^[A-Z] | xargs) docker stack deploy --compose-file docker-compose.yml snipsnlu
