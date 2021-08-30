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
RUN pip install shapely



RUN apt-get -y install libgsl-dev pkg-config libhdf5-serial-dev libboost-all-dev python-dev && \
    mkdir -p /usr/local/lib/SQuIDS && \
    cd /usr/local/lib/SQuIDS && \
    git clone https://github.com/jsalvado/SQuIDS.git . && \
    ./configure && make && make && \
    mkdir -p /usr/local/lib/nuSQuIDS && \
    cd /usr/local/lib/nuSQuIDS && \
    git clone https://github.com/arguelles/nuSQuIDS.git . && \
    ./configure --with-python-bindings --with-squids=/usr/local/lib/SQuIDS && \
    cp /opt/conda/lib/libpython3.8.so /usr/local/lib && \
    make && make install && LD_LIBRARY_PATH=/opt/conda/lib/:$LD_LIBRARY_PATH make python && make python-install

RUN PATH=/usr/local/lib/nodejs/node-v14.17.0-linux-x64/bin:$PATH jupyter labextension install jupyterlab-plotly

RUN mkdir -p /usr/local/lib/MCEq && cd /usr/local/lib/MCEq && \
    git clone -b next_1_3_X https://github.com/afedynitch/MCEq.git . && \
    pip install .[CUDA]
COPY mceq_db_lext_dpm191_v131.h5 /opt/conda/lib/python3.8/site-packages/MCEq/data/
#RUN pip install MCEq[CUDA]

RUN python -c "from MCEq.core import MCEqRun"
CMD tensorboard --port 8008 --logdir=/app/runs --bind_all & \
    PATH=/usr/local/lib/nodejs/node-v14.17.0-linux-x64/bin:$PATH \
    PYTHONPATH=$PYTHONPATH:/opt/PROPOSAL/build/src/pyPROPOSAL:/usr/lib/nuSQuIDS/resources/python/bindings/ \
    jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/app
