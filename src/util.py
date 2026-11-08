from .settings import Settings
from .models import ConfigurationPrompt
from .chat import create_chat


ROLE2TEMPLATE: dict[str, str] = {
    "System": "system",
    "User": "human",
    "Ai": "ai",
}


def create_chat_(configuration: ConfigurationPrompt):
    return create_chat(Settings.PROVIDER, configuration.model)
