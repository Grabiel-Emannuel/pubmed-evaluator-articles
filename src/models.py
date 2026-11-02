from typing import Literal

from pydantic import BaseModel


class Prompt(BaseModel):
    role: Literal["System", "User", "Ai"]
    instruction: str


class ConfigurationPrompt(BaseModel):
    model: str
    prompts: list[Prompt]


class ArticleInformation(BaseModel):
    title: str
    abstract: str
