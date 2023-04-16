FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV LANG C.UTF-8

WORKDIR /root/app
# ab相关基础配置
# COPY ./docker_base/sources.list /etc/apt/sources.list
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-get update \
    && apt-get install -y \
    curl \
    vim \
    iputils-ping \
    git \
    locales \
    logrotate \
    nginx \
    cron \
    && apt-get clean
# 设置pip使用阿里云镜像
COPY ./docker_base/.pip.conf /root/.pip/pip.conf

# 设置lograte的配置
COPY ./docker_base/gunicorn /etc/logrotate.d/gunicorn

# 启用lograte的定时任务
COPY ./docker_base/gunicorn.cron /etc/cron.d/gunicorn.cron
COPY ./docker_base/crond /etc/pam.d/crond

RUN echo "abcdefghi"

# 启动命令
COPY ./docker_base/launch.sh /root/launch.sh

# python文件加密脚本
COPY ./docker_base/encrypt.sh ./encrypt.sh

# 数据文件加密脚本
COPY ./docker_base/crypto.sh ./crypto.sh

RUN chmod 0644 /etc/cron.d/gunicorn.cron
RUN chmod 777 /root/launch.sh

RUN crontab /etc/cron.d/gunicorn.cron
RUN touch /var/log/cron.log

# 拷贝nginx配置
COPY ./docker_base/nginx_default.conf /etc/nginx/sites-available/default
COPY ./docker_base/nginx.conf /etc/nginx/nginx.conf


# 安装算法依赖包
COPY ./setup/base.txt setup/base.txt
RUN pip install  --no-cache-dir -r setup/base.txt

COPY ./setup/requirements.txt setup/requirements.txt
RUN pip install  --no-cache-dir -r setup/requirements.txt
COPY ./setup/ab-3.3.0.0-py3-none-any.whl setup/ab-3.3.0.0-py3-none-any.whl
RUN pip install setup/ab-3.3.0.0-py3-none-any.whl
# 源代码等
COPY ./ ./

# 必须创建日志目录
RUN mkdir -p logs

# 加密python代码
RUN bash ./encrypt.sh

# 加密数据文件
RUN bash ./crypto.sh
# 加密结束后，删除加密密钥
RUN rm -rf ./crypto.sh

# 启动服务。本地调试时，可以覆盖启动命令
ENTRYPOINT ["/root/launch.sh"]
CMD ["prod"]

