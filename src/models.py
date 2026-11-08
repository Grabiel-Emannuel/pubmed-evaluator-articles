from typing import Literal

from pydantic import BaseModel, Field


class Prompt(BaseModel):
    role: Literal["System", "User", "Ai"]
    instruction: str


class ConfigurationPrompt(BaseModel):
    model: str
    think: int = Field(default=1, ge=0, le=1)
    prompts: list[Prompt]


class ArticleInformation(BaseModel):
    title: str
    abstract: str
