sudo mkdir /snips-data

# Cloning mnemoose only if it does not exist
if [ ! -d "snips-nlu-parse" ]; then
  git clone git@github.com:SmartlyAI/snips-nlu-parse.git
  cd snips-nlu-parse && git checkout $1 && cd ..
else
  echo 'Updating snips-nlu-parse repository'
  cd snips-nlu-parse &&  git fetch && git checkout $1 && git pull origin $1 && cd ..
fi

# SERVICES

# Cloning atlas only if it does not exist
if [ ! -d "snips-nlu-train" ]; then
 git clone git@github.com:SmartlyAI/snips-nlu-train.git
 cd snips-nlu-train && git checkout $1 && cd ..
else
  echo 'Updating snips-nlu-train repository'
  cd snips-nlu-train &&  git fetch && git checkout $1 && git pull origin $1 && cd ..
fi

docker-compose build && docker-compose up -d
