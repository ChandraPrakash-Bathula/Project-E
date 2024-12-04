# official python runtime as base image
FROM python:3.11-slim

# working directory inside the container
WORKDIR /app

# copy the requirements.txt file into the work dir
COPY requirements.txt .

# install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# copy all from here onto /app dir
COPY . .

# Expose the port to run fastapi pn
EXPOSE 8000

# command to run the fastapi app using uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
