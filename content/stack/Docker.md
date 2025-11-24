---
title: Docker 사용법
publish: false
tags:
  - Tool-stack
date: 2024-07-05
---




# 주요 명령어

```bash
sudo docker run -it -d --gpus all --ipc=host -v {붙일 디렉토리}:{컨테이너 속 디렉토리} --name {이름} {이미지 풀 태그}
```


## 예시
```bash
sudo docker run -it -d --gpus all --ipc=host -v /home/idsh/workspace/bayesopt:/workspace/KOLON-BO --name KOLON 163.152.40.183:5005/kolon:v0.0.2
```

docker image 받아올땐 `pull` 명령어를 쓴다
## 예시
```bash
docker pull 163.152.40.183:5005/kolon:v0.0.2
```