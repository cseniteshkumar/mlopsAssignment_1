FROM python:3.10-slim

WORKDIR /app

# copy requirements and install (no cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# copy source
COPY . /app
# expose app port
EXPOSE 8080

# default command: run the FastAPI app with uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
