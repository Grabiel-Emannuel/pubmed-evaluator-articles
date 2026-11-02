from os import getenv

from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROVIDER: str | None = getenv("PROVIDER")
    OLLAMA_HOST: str | None = getenv("OLLAMA_HOST")
