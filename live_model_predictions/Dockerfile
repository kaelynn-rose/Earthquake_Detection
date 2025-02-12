FROM public.ecr.aws/lambda/python:3.12

# Define build-time variables
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends wget python3 python3-pip bash curl && \
    apt-get clean && \
    pip3 install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir awscli

# Install Miniconda
RUN curl -sSLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -f -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda init bash \
    && apt-get clean

# Set the path for Conda
ENV PATH="/opt/conda/bin:$PATH"

# Set Mamba env name
ENV MAMBA_ENV="live-predictions-lambda"

# Install Mamba for faster package management
RUN conda install -c conda-forge mamba -y

# Copy the environment.yaml file into the container
COPY env.yaml /tmp/env.yaml

# Create the Conda environment from the environment.yaml file
RUN /bin/bash -c "mamba env create -f /tmp/env.yaml && mamba clean -a && echo 'source activate ${MAMBA_ENV}' >> ~/.bashrc"

# Set working directory
WORKDIR /home/app

COPY lambda.py /home/app

CMD ["lambda.lambda_handler"]