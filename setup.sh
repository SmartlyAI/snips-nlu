# Swarm alreay cerated with docker swarm init on azure
docker swarm init

# Create registry which will contains all images for smartly services
docker service create --name registry-smartly-swarm --publish 5000:5000 --constraint 'node.role == manager' --detach=false registry:2

# Create volumes
docker volume create --name=rabbitmq

# Cloning hestia only if it does not exist
if [ ! -d "hestia" ]; then
  git clone git@github.com/SmartlyAI/hestia.git
  cd hestia && git checkout snips && cd ..
else
  echo 'Updating hestia repository'
  cd hestia &&  git fetch && git checkout snips && git pull origin snips && cd ..
fi

# Cloning snips-nlu-parse only if it does not exist
if [ ! -d "snips-nlu-parse" ]; then
  git clone git@github.com:SmartlyAI/snips-nlu-parse.git
  cd snips-nlu-parse && git checkout $1 && cd ..
else
  echo 'Updating snips-nlu-parse repository'
  cd snips-nlu-parse &&  git fetch && git checkout $1 && git pull origin $1 && cd ..
fi

# Cloning snips-nlu-train only if it does not exist
if [ ! -d "snips-nlu-train" ]; then
 git clone git@github.com/SmartlyAI/snips-nlu-train.git
 cd snips-nlu-train && git checkout $1 && cd ..
else
  echo 'Updating snips-nlu-train repository'
  cd snips-nlu-train &&  git fetch && git checkout $1 && git pull origin $1 && cd ..
fi

# Build stack
privatekey=$(cat ~/.ssh/id_rsa)
publickey=$(cat ~/.ssh/id_rsa.pub)
SSH_PRIVATE_KEY="$privatekey" SSH_PUBLIC_KEY="$publickey" docker-compose build

# Push and launch snips_nlu stack
docker-compose push && env $(cat .env | grep ^[A-Z] | xargs) docker stack deploy --compose-file docker-compose.yml snipsnlu
