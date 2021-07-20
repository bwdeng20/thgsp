#!/bin/bash

if [ "${RUNNER_OS}" = "Linux" ]; then
  sudo add-apt-repository ppa:ubuntu-toolchain-r/test --yes
  sudo apt update
  sudo apt install gcc-7 g++-7 --yes
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                           --slave /usr/bin/g++ g++ /usr/bin/g++-7
  sudo update-alternatives --config gcc
  gcc --version
  g++ --version
else
      echo "Don't consider other systems at present!"
      exit 1
fi
