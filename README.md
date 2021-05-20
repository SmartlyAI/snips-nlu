# snips-nlu

## Service version numbers for docker and docker-compose:
- Docker Version: 20.10.5
- Docker-compose Version: 1.28.5

# To deploy from scratch on production:

## 1- Install/launch App stack:
- First, you need to clone this repo with the command line `git clone`
- Second, you need to add .env file which contains the environment variable used by docker-compose.
- Third, launch the _setup.sh_ script with the following command `bash setup.sh`.

## 2- Configure Nginx:
- launch the _nginx.setup.sh_ script with the following command `sh nginx.setup.sh`.

# To update the NLU application:
- First, pull the commits of the repo smartly-swarm-azure `git pull`.
- If necessary: [Second, add new env variables in the file .env (Rarley)]
- Third, load image from tar file : `bash save-load-docker-images.sh load`. The script must be executed in the same directory where docker images are stored.
- Finally, re-launch the stack `env $(cat .env | grep ^[A-Z] | xargs) docker stack deploy --compose-file docker-compose.yml snipsnlu`.

