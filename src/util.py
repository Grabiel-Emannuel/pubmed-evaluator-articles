from .settings import Settings
from .models import ConfigurationPrompt
from .chat import create_chat

def create_chat_(configuration: ConfigurationPrompt):
    return create_chat(Settings.PROVIDER, configuration.model)
