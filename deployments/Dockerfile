FROM tensorflow/serving:2.17.0

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
ENV MAMBA_ENV="earthquake-detection-serving"

# Install Mamba for faster package management
RUN conda install -c conda-forge mamba -y

# Copy the environment.yaml file into the container
COPY deployments/env.yaml /tmp/env.yaml

# Create the Conda environment from the environment.yaml file
RUN /bin/bash -c "mamba env create -f /tmp/env.yaml && mamba clean -a && echo 'source activate ${MAMBA_ENV}' >> ~/.bashrc"

# Set working directory
WORKDIR /home/app

COPY deployments/entrypoint.sh entrypoint.sh
COPY deployments/serving serving
COPY earthquake_detection earthquake_detection

# Download the saved ML models from S3
RUN aws s3 cp s3://earthquake-detection/models/signal_classification_model/ /home/app/models/model_1 --recursive
RUN aws s3 cp s3://earthquake-detection/models/earthquake_magnitude_prediction_model/ /home/app/models/model_2 --recursive

# Expose port 5000 for API
EXPOSE 5000

# Run the entrypoint script
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]