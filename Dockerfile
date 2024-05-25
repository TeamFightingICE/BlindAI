FROM ubuntu:20.04

WORKDIR /app

# Install miniconda3
ENV CONDA_HOME=/opt/conda
RUN apt-get update && apt-get install -y wget \
    && wget -O ./miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-py312_24.4.0-0-Linux-x86_64.sh \
    && bash ./miniconda3.sh -b -u -p ${CONDA_HOME} \
    && rm ./miniconda3.sh

# Install python dependencies
COPY ./environment.yml .
RUN ${CONDA_HOME}/bin/conda env create -n ice -f environment.yml \
    && ${CONDA_HOME}/envs/ice/bin/pip install -i https://test.pypi.org/simple/ pyftg==2.2b1 \
    && ${CONDA_HOME}/envs/ice/bin/pip cache purge \
    && ${CONDA_HOME}/bin/conda clean -y -a \
    && rm ./environment.yml

COPY ./*.py .

ENTRYPOINT [ "${CONDA_HOME}/envs/ice/bin/python", "main.py" ]