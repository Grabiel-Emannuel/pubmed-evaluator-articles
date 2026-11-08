from os import getenv

from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROVIDER: str | None = getenv("PROVIDER")
    OLLAMA_HOST: str | None = getenv("OLLAMA_HOST")
    NLLB_API_URL: str = getenv("NLLB_API_URL", "http://192.168.122.1:7860").rstrip("/")
    NLLB_TIMEOUT_SECONDS: float = float(getenv("NLLB_TIMEOUT_SECONDS", "120"))
