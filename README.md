# Final-Project
# How to Deploy Streamlit app on EC2 instance

## 1. Login with your AWS console and launch an EC2 instance

## 2. Run the following commands

### Note: Do the port mapping to this port:- 8501

```bash
sudo apt update
```

```bash
sudo apt upgrade -y
```

```bash
sudo apt install git curl unzip tar make vim wget python3-pip python3-venv build-essential python3-dev -y
```

```bash
git clone "YOUR-REPOSITORY-URL"
cd YOUR-REPO-NAME
```

```bash
python3 -m venv env
source env/bin/activate
```

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

```bash
#Temporary running
streamlit run final.py
```

```bash
#Permanent running
nohup streamlit run final.py
```

Note: Streamlit runs on this port: 8501



