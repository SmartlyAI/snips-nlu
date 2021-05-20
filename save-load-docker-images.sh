#!/usr/bin/env bash

# Script to (selectively) save/load multiple Docker images to/from a directory.
# Run ./save-load-docker-images.sh for help.

set -e

directory=$PWD


help () {
    echo
    echo "\
Usage: save  - Save all Docker images to a directory (equal to \$PWD)
       load  - Find all saved images (.tar.gz) in a directory then import to Docker (equal to \$PWD)"
    echo
}

get-image-field() {
  local imageId=$1
  local field=$2
  : ${imageId:? required}
  : ${field:? required}

  docker images --no-trunc | sed -n "/${imageId}/s/  */ /gp" | cut -d " " -f $field
}

get-image-name() {
  get-image-field $1 1
}

get-image-tag() {
  get-image-field $1 2
}

save-all-images() {
  local ids=$(docker images --no-trunc -q)
  local name safename tag

  for id in $ids; do
    name=$(get-image-name $id)
    tag=$(get-image-tag $id)

    if [[ $name =~ / ]]; then
       dir=${name%/*}
       mkdir -p "$directory/$dir"
    fi

    echo "Saving $name:$tag ..."
    docker save $name:$tag | gzip > "$directory/$name.$tag.tar.gz"
  done
}

load-all-images() {
  local name safename noextension tag

  for image in $(find "$directory" -name \*.tar.gz); do
    echo "Loading $image ..."
    docker load -i $image
  done
}

case $1 in
    save)
      save-all-images
    ;;
    load)
      load-all-images
    ;;
    *)
        help
    ;;
esac

exit 0