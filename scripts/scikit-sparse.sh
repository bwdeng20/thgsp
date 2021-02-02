#!/bin/bash

if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
sudo apt-get install python-scipy libsuitesparse-dev
fi