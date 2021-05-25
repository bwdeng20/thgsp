#!/bin/bash

METIS=metis-5.1.0
export WITH_METIS=1

wget -nv http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/${METIS}.tar.gz
tar -xvzf ${METIS}.tar.gz
cd ${METIS} || exit
sed -i.bak -e 's/IDXTYPEWIDTH 32/IDXTYPEWIDTH 64/g' include/metis.h

if [ "${RUNNER_OS}" != "Windows" ]; then
  make config
  make
  sudo make install
else
  echo "Don't consider other operation systems at present!"
fi

cd ..