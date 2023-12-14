# BurntBase Pytorch API 🔥

FASTAPI server that powers burntbase detections

## Local Development 🔧

**Create a New Enviroment**
```
conda create --name burntbasepyapi python=3.9
```

**Activate New Enviroment**

```
conda activate burntbasepyapi
```

**Install Requirements**
```
pip install -r requirements.txt
```

**Run the App**
```
cd api && uvicorn api:app --reload  
```

The app should be up and being served at **localhost8000** by default

## Running On an EC2 Instance 🏃‍♂️

Create any EC2 instance with a GPU. Use the **Deep Learning AMI GPU PyTorch 2.0.1 (Amazon Linux 2) 20230627** deeplearning image for pytorch provided by AWS. This should have pytorch installed correctly and make everything smooth.


Activate the pytorch enviroment:

```
source activate pytorch
```

Run uvicorn. Note the host has to be **0.0.0.0** for it to work properly and serve requests

```
uvicorn api:app --port 5001 --host 0.0.0.0
```
