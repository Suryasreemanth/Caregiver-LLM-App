# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the Docker container
WORKDIR /app

# Install any necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file to the working directory
COPY requirements.txt /app/

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . /app/

# Expose the port on which Streamlit will run
EXPOSE 8501

# Health check to ensure the container is running correctly
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run your application
ENTRYPOINT ["streamlit", "run", "miller-lab-agent.py", "--server.port=8501", "--server.address=0.0.0.0"]
