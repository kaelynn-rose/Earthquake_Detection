# Debian-based Python image that uses apt-get package manager
FROM public.ecr.aws/lambda/python:3.8

# Set the working directory inside the container
WORKDIR /home/app

# Copy the lambda function code and environment requirements into the container
COPY lambda_function.py ./
COPY requirements.txt ./

# Install the dependencies
RUN python3 -m pip install -r requirements.txt

# Set the CMD to the Lambda handler function
CMD ["/home/app/lambda_function.lambda_handler"]