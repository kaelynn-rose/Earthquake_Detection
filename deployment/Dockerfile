FROM tensorflow/serving:2.17.0

# Define build-time variables
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

# Install Miniconda
RUN apt-get update && apt-get install -y \
    python3-pip \
    curl \
    awscli \
    && curl -sSLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -f -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda init bash \
    && apt-get clean

# Set the path for Conda
ENV PATH=/opt/conda/bin:$PATH

# Copy the environment.yaml file into the container
#COPY environment.yaml environment.yaml

# Create the Conda environment from the environment.yaml file
RUN conda env create -f environment.yaml && conda clean -a

CMD pwd && ls

# Set working directory
WORKDIR /app

# Copy the entrypoint script
#COPY deployment/entrypoint.sh entrypoint.sh

# Download the saved ML models from S3
RUN aws s3 cp s3://earthquake-detection/models/signal_classification_model/ models/model_1 --recursive
RUN aws s3 cp s3://earthquake-detection/models/earthquake_magnitude_prediction_model/ models/model_2 --recursive

# Expose the necessary ports for the API and ML models
EXPOSE 5000 8501 8502

# Run the entrypoint script
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]