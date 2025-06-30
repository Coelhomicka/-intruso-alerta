import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# ROOT_DIR = pasta raiz do seu repositório
ROOT_DIR = Path(__file__).resolve().parents[2]

# carrega variáveis de ambiente
load_dotenv(ROOT_DIR / ".env")

def load_config():
    # lê o config.yml (YAML puro de configuração)
    cfg_path = ROOT_DIR / "config.yml"
    cfg_file = yaml.safe_load(cfg_path.read_text())

    return {
        "lock_threshold": int(os.getenv("LOCK_THRESHOLD", 10)),
        "save_dir": Path(os.getenv("SAVE_DIR", str(ROOT_DIR / "intruder_photos"))),
        "smtp": {
            "host": os.getenv("SMTP_HOST"),
            "port": int(os.getenv("SMTP_PORT", 587)),
            "user": os.getenv("SMTP_USER"),
            "password": os.getenv("SMTP_PASS"),
            "to_addr": os.getenv("ALERT_EMAIL"),
        },
        # aqui só passamos o caminho do modelo (trainer.yml)
        "trainer": {
            "output": str(ROOT_DIR / cfg_file["model_path"])
        },
    }
