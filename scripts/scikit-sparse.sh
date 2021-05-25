#!/bin/bash

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
sudo apt-get install python-scipy libsuitesparse-dev
else
    echo "thgsp only supports Linux at present!"
    exit 1
fi