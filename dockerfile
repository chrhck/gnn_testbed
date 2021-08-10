FROM pytorch/pytorch:latest as base
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install \
      build-essential \
      gcc \
      git \
      libssl-dev \
      python3-dev \
      wget && \
    rm -rf /var/lib/apt/lists/* && \
    wget https://github.com/Kitware/CMake/releases/download/v3.20.3/cmake-3.20.3.tar.gz && \
        tar -zxvf cmake-3.20.3.tar.gz && \
        cd cmake-3.20.3 && ./bootstrap && make -j4 && make install && \
        pip install conan && \
        conan profile new default --detect && conan profile update settings.compiler.libcxx=libstdc++11 default && \
        mkdir /opt/PROPOSAL && cd /opt/PROPOSAL && \
        git clone https://github.com/tudo-astroparticlephysics/PROPOSAL.git . && \
        mkdir -p build && cd build && conan install .. -o with_python=True && \
        conan build ..

FROM base as base_tables
COPY build_proposal_tables.py /opt/PROPOSAL/build_proposal_tables.py
RUN PYTHONPATH=$PYTHONPATH:/opt/PROPOSAL/build/src/pyPROPOSAL python3 /opt/PROPOSAL/build_proposal_tables.py

FROM base_tables as node
RUN wget https://nodejs.org/dist/v14.17.0/node-v14.17.0-linux-x64.tar.xz && \
    mkdir -p /usr/local/lib/nodejs && \
    tar -xJvf node-v14.17.0-linux-x64.tar.xz -C /usr/local/lib/nodejs
RUN apt-get update && \
    apt-get -y install npm

FROM node as pip_stuff
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-geometric jupyterlab awkward numba seaborn tqdm ipywidgets aquirdturtle_collapsible_headings plotly tensorboard matplotlib_inline
RUN pip install shapely MCEq[CUDA]
RUN apt-get -y install libgsl-dev pkg-config && \
    mkdir -p /usr/local/lib/SQuIDS && \
    cd /usr/local/lib/SQuIDS && \
    git clone https://github.com/jsalvado/SQuIDS.git . && \
    ./configure && make && make install
RUN PATH=/usr/local/lib/nodejs/node-v14.17.0-linux-x64/bin:$PATH jupyter labextension install jupyterlab-plotly

CMD tensorboard --port 8008 --logdir=/app/runs --bind_all & \
    PATH=/usr/local/lib/nodejs/node-v14.17.0-linux-x64/bin:$PATH \
    PYTHONPATH=$PYTHONPATH:/opt/PROPOSAL/build/src/pyPROPOSAL \
    jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/app
