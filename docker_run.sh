
#!/bin/bash
username=yoshimura
data_path="/mnt/poplin/share/2025/users/yoshimura"

docker run --gpus all \
        -it --rm \
        -u $(id -u $username):$(id -g $username) \
        --name ${username}gaussian_splatting \
        -w /working \
        -v $PWD/../gaussian_splatting/:/working \
        -v $data_path/:/data \
        repo-luna.ist.osaka-u.ac.jp:5000/${username}/gaussian_splatting:latest \
        bash
