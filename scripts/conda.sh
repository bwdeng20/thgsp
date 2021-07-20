if [ "${TRAVIS_OS_NAME}" = "linux" ]; then
  wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  chmod +x miniconda.sh
  ./miniconda.sh -b
  PATH=/home/travis/miniconda3/bin:${PATH}
else
  echo "Don't consider windows at present!"

fi

conda update --yes conda
conda create --yes -n test python="${PYTHON_VERSION}"