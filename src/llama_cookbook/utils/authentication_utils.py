import os
from huggingface_hub import login

def huggingface_login(token_path: str = "keys/huggingface_token.txt"):
    """Logs into Hugging Face using a token stored in a file."""
    try:
        with open(token_path, 'r') as file:
            token = file.read().strip()
            login(token=token)
            print("Successfully logged into Hugging Face.")
    except FileNotFoundError:
        print(f"Token file not found at {token_path}. Please provide a valid path.")
    except Exception as e:
        print(f"An error occurred during login: {e}")
        

def ssh_env_settings():
    # for gcs
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')
    # for proxy
    os.environ["http_proxy"] = "http://sysproxy.wal-mart.com:8080"
    os.environ["https_proxy"] = "http://sysproxy.wal-mart.com:8080"
    os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
    os.environ["NO_PROXY"] = ".walmart.com,.wal-mart.com,.walmart.net,.wal-mart.net,localhost,127.0.0.1,api.wandb.ai,.wandb.ai"