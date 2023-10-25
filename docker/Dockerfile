FROM tensorflow/tensorflow:latest-gpu

ARG user_id
ARG root_psw="12345678"
ARG user_psw="ok"
ARG user_name=user

# Installs the necessary pkgs.
RUN \
	echo "**** packages installation ****" \
		&& apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub \
		&& apt-get update && apt-get install -y \
			vim \
            build-essential \
            cmake \
            libopencv-dev \
            libjpeg-dev \
            libpng-dev \
            libglew-dev \
            libpthread-stubs0-dev \
			git \
			virtualenv \
			time \
			sudo \
			wget \
			nano \
            python3.8-venv \
            libboost-program-options-dev \
            libboost-filesystem-dev \
            libboost-graph-dev \
            libboost-regex-dev \
            libboost-system-dev \
            libboost-test-dev \
            libeigen3-dev \
            libsuitesparse-dev \
            libfreeimage-dev \
            libgoogle-glog-dev \
            libgflags-dev \
            libglew-dev \
            qtbase5-dev \
            libqt5opengl5-dev \
            libcgal-dev \
            libcgal-qt5-dev \
            libmetis-dev \
            libflann-dev \
            libatlas-base-dev \
            libsuitesparse-dev \
	&& echo "**** python pip update ****" \
		&& /usr/bin/python3 -m pip install --upgrade pip \
	&& echo "**** aliases for l and ll commands creation ****" \
		&& echo -e 'ls --color=auto "$@"' > /usr/bin/l \
		&& echo -e 'ls -lah --color=auto "$@"' > /usr/bin/ll \
    	&& chmod +x /usr/bin/ll /usr/bin/l \
	&& echo "**** history-search-backward by pressing F8 ****" \
		&& sed -i 's/# "\\e\[5~": history-search-backward/"\\e\[19~": history-search-backward/' /etc/inputrc \
	&& echo "**** root password creation ****" \
		&& echo "root:${root_psw}" | chpasswd \
	&& echo "**** user creation ****" \
		&& useradd -m -s /usr/bin/bash -u ${user_id} -G sudo ${user_name} \
		&& echo "${user_name}:${user_psw}" | chpasswd \
		&& chown -R ${user_name}:${user_name} /home/${user_name}/ \
		&& mkdir /home/${user_name}/workspace/ \
        && chown -R user:user /home/${user_name}/workspace


USER ${user_name} 
WORKDIR /home/${user_name}/


RUN git config --global http.sslverify false
RUN cd /home/${user_name}/workspace && git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting && cd gaussian-splatting && ./docker/initialize.sh

