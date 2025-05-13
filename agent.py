import contextlib
import io
import logging
import os
logger = logging.getLogger(__name__)
from models import GoogleModelID, OpenRouterModelID
from settings import Settings
from smolagents import LiteLLMModel, CodeAgent
from smolagents import GoogleSearchTool, VisitWebpageTool, FinalAnswerTool
from smolagents.local_python_executor import BASE_PYTHON_TOOLS
from tools import GetTaskFileTool, VideoUnderstandingTool, AudioUnderstandingTool
from tools import ChessBoardFENTool, BestChessMoveTool, ConvertChessMoveTool


# Base tools may use these to process files
BASE_PYTHON_TOOLS["open"] = open
BASE_PYTHON_TOOLS["os"] = os
BASE_PYTHON_TOOLS["io"] = io
BASE_PYTHON_TOOLS["contextlib"] = contextlib
BASE_PYTHON_TOOLS["exec"] = exec

class ResearchAgent:
    def __init__(self, settings: Settings):
        self.agent = CodeAgent(
            name="researcher",
            description="Searches the web, works with files, and answers questions for you. Give it your query as an argument.",
            add_base_tools=False,
            tools=[GoogleSearchTool("serper"),
                   VisitWebpageTool(max_output_length=100000),
                   VideoUnderstandingTool(settings, GoogleModelID.GEMINI_2_0_FLASH),
                   AudioUnderstandingTool(settings, GoogleModelID.GEMINI_2_0_FLASH)
                   ],
            additional_authorized_imports=[
                "unicodedata",
                "stat",
                "datetime",
                "random",
                "pandas",
                "itertools",
                "math",
                "statistics",
                "queue",
                "time",
                "collections",
                "re",
                "os"
            ],
            max_steps=10,
            verbosity_level=1,
            model=LiteLLMModel(
                model_id=OpenRouterModelID.GPT_O4_MINI_HIGH,
                api_key = settings.openrouter_api_key.get_secret_value(),
                temperature=0.0, timeout=180
            )
        )

class ChessAgent:
    def __init__(self, settings: Settings):
        self.agent = CodeAgent(
            name="chess_player",
            description="Makes a chess move. Give it a query including board image filepath and player turn (black or white).",
            add_base_tools=False,
            tools=[ChessBoardFENTool(),
                   BestChessMoveTool(settings),
                   ConvertChessMoveTool(settings, OpenRouterModelID.GPT_O4_MINI),
                   ],
            additional_authorized_imports=[
                "unicodedata",
                "stat",
                "datetime",
                "random",
                "pandas",
                "itertools",
                "math",
                "statistics",
                "queue",
                "time",
                "collections",
                "re",
                "os"
            ],
            max_steps=10,
            verbosity_level=1,
            model=LiteLLMModel(
                model_id=OpenRouterModelID.GPT_O4_MINI,
                api_key = settings.openrouter_api_key.get_secret_value(),
                temperature=0.0, timeout=180
            )
        )

class ManagerAgent:
    def __init__(self, settings: Settings):
        self.researcher = ResearchAgent(settings).agent
        self.chess_player = ChessAgent(settings).agent
        self.agent = CodeAgent(
            tools=[GetTaskFileTool(settings), FinalAnswerTool()],
            model=LiteLLMModel(
                model_id=OpenRouterModelID.GPT_O4_MINI,
                api_key = settings.openrouter_api_key.get_secret_value(),
                temperature=0.0, timeout=180
            ),
            managed_agents=[self.researcher, self.chess_player],
        )
        # print("BasicAgent initialized.")
    def __call__(self, question: str) -> str:
        logger.info(f"Agent received question (first 50 chars): {question[:50]}...")
        final_answer = self.agent.run(question)
        logger.info(f"Agent returning fixed answer: {final_answer}")
        return final_answer 