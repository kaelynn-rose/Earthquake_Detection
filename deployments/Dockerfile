FROM tensorflow/serving:2.17.0

# Define build-time variables
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

#COPY deployment/requirements.txt requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends wget python3 python3-pip bash curl && \
    apt-get clean && \
    pip3 install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir awscli
    #pip3 install --no-cache-dir -r requirements.txt

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
COPY env.yaml /tmp/env.yaml

# Create the Conda environment from the environment.yaml file
#RUN mamba env create -f /tmp/env.yaml && conda clean -a
RUN /bin/bash -c "mamba env create -f /tmp/env.yaml && mamba clean -a && echo 'source activate ${MAMBA_ENV}' >> ~/.bashrc"

# Set working directory
WORKDIR /home/app

# Copy the entrypoint script
COPY entrypoint.sh entrypoint.sh

# Download the saved ML models from S3
RUN aws s3 cp s3://earthquake-detection/models/signal_classification_model/ /models/model_1 --recursive
RUN aws s3 cp s3://earthquake-detection/models/earthquake_magnitude_prediction_model/ /models/model_2 --recursive

# Expose the necessary ports for the API and ML models
EXPOSE 5000 8501 8502

# Run the entrypoint script
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]