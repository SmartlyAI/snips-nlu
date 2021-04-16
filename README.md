# snips-nlu

## Service version numbers for docker and docker-compose:
- Docker Version: 20.10.5
- Docker-compose Version: 1.28.5

## To deploy from scratch on production:
- First, you need to clone this repo with the command line `git clone`
- Second, you need to add .env file which contains the environment variable used by docker-compose.
- Third, launch the _setup.sh_ script with the following command `bash setup.sh magenta-develop`.

## To install Nginx/Amplify:
- launch the _nginx.setup.sh_ script with the following command `env $(cat .env | grep ^[A-Z] | xargs) sh nginx.setup.sh`.