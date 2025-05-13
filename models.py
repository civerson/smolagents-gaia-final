# from enum import StrEnum
from pydantic import BaseModel, ConfigDict


class GoogleModelID():
  GEMINI_2_0_FLASH = "gemini-2.0-flash"
  GEMINI_2_5_FLASH_PREVIEW = "gemini-2.5-flash-preview"

class OpenRouterModelID():
  QWEN_3_14B_FREE = "openrouter/qwen/qwen3-14b:free"
  GPT_4_1_MINI = "openrouter/openai/gpt-4.1-mini"
  GPT_O4_MINI = "openrouter/openai/o4-mini"
  GPT_O4_MINI_HIGH = "openrouter/openai/o4-mini-high"
  GROK_3_MINI_BETA = "openrouter/x-ai/grok-3-mini-beta"
  GROK_3_BETA = "openrouter/x-ai/grok-3-beta"
  
class Question(BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
    task_id: str
    question: str
    file_name: str

class Answer(BaseModel):
    task_id: str
    answer: str

class QuestionAnswerPair(BaseModel):
    task_id: str
    question: str
    answer: str
    
    def get_answer(self) -> dict[str, str]:
        return {"task_id": self.task_id, "submitted_answer": self.answer}

class Results(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    username: str
    score: int
    correct_count: int
    total_attempted: int
    message: str
    timestamp: str
