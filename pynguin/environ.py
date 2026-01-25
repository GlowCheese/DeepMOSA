import os

import dotenv

from libs.custom_logger import getLogger

dotenv.load_dotenv()
logger = getLogger(__name__)

PYNGUIN_DANGER_AWARE = os.getenv("PYNGUIN_DANGER_AWARE")

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
