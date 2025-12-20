import os

import dotenv

from pynguin.utils.custom_logger import getLogger

dotenv.load_dotenv()
logger = getLogger(__name__)

PYNGUIN_DANGER_AWARE = os.getenv("PYNGUIN_DANGER_AWARE")

CHATGPT_BASE_URL = "https://api.openai.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

OPENAI_API_KEY: str | None = os.getenv("DEEPSEEK_API_KEY")
