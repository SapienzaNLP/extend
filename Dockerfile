FROM nvidia/cuda:11.1-devel-ubuntu20.04

WORKDIR /root

# install utilities

RUN \
    DEBIAN_FRONTEND="noninteractive" apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y wget git && \
    mkdir .ssh
EXPOSE 22001
EXPOSE 22002

# install conda
RUN \
    wget -O miniconda.sh "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh
ENV PATH=/root/miniconda3/bin:${PATH}
RUN conda update -y conda && conda init

# setup env
WORKDIR /extend
COPY . .
RUN \
    bash -c "source ~/miniconda3/etc/profile.d/conda.sh && printf 'extend\n3.8\n11.1\n' | bash setup.sh"

# standard cmd
CMD "./demo.sh"
