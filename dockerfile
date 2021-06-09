FROM pytorch/pytorch:latest
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install gcc git build-essential python3-dev wget libssl-dev && \
    rm -rf /var/lib/apt/lists/*
#protobuf-compiler, python-dev
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.3/cmake-3.20.3.tar.gz && \
    tar -zxvf cmake-3.20.3.tar.gz && \
    cd cmake-3.20.3 && ./bootstrap && make -j4 && make install
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-geometric jupyterlab conan awkward numba seaborn
RUN conan profile new default --detect && conan profile update settings.compiler.libcxx=libstdc++11 default
RUN mkdir /opt/PROPOSAL && cd /opt/PROPOSAL && \
    git clone https://github.com/tudo-astroparticlephysics/PROPOSAL.git . && \
    mkdir -p build && cd build && conan install .. -o with_python=True && \
    conan build ..
CMD PYTHONPATH=$PYTHONPATH:/opt/PROPOSAL/build/src/pyPROPOSAL jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root --notebook-dir=/app
