
# Specify the parent image from which we build
FROM python:3.9 as data_dump

# Set the working directory
WORKDIR /app

# Copy files from your host to your current working directory
COPY requirements.txt requirements.txt

# Build the application with cmake
RUN apt-get dist-upgrade && \
    apt-get install -y git && \
    pip --no-cache-dir install --upgrade pip && \
    pip --no-cache-dir install -r requirements.txt && \
    apt-get remove -y git && \
    rm -rf /var/lib/apt/lists/*

# Install huggingface-cli
# RUN pip install --upgrade huggingface-hub && pip install git-lfs

# Login to Hugging Face Hub (you'll need to provide your API key)
# ARG HUGGINGFACE_TOKEN
# RUN huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential


ENV PYTHONFAULTHANDLER=1

COPY . .

ENTRYPOINT [ "python3" ]

