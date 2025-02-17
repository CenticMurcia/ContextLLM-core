FROM python:3.10.7-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libreadline-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN wget https://www.sqlite.org/2022/sqlite-autoconf-3370200.tar.gz \
    && tar -xvzf sqlite-autoconf-3370200.tar.gz \
    && cd sqlite-autoconf-3370200 \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm -rf sqlite-autoconf-3370200*
RUN ldconfig
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
CMD ["streamlit","run", "/app/front.py"]
