import os

import dotenv

from vendor.custom_logger import getLogger

dotenv.load_dotenv()
logger = getLogger(__name__)


PYNGUIN_DANGER_AWARE = os.getenv("PYNGUIN_DANGER_AWARE")


DEFAULT_MODEL = os.environ["DEFAULT_MODEL"]

CHATGPT_BASE_URL = "https://api.openai.com/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

OPENAI_API_KEY: str | None = os.getenv(f'{DEFAULT_MODEL}_API_KEY')

if OPENAI_API_KEY is None:
    logger.warning(f'API_KEY for {DEFAULT_MODEL} is missing')

if DEFAULT_MODEL == "DEEPSEEK":
    OPENAI_CHAT_MODEL = "deepseek-chat"
    OPENAI_COMP_MODEL = "deepseek-chat"
    OPENAI_BASE_URL = DEEPSEEK_BASE_URL

elif DEFAULT_MODEL == "CHATGPT":
    OPENAI_CHAT_MODEL = "gpt-4.1-mini-2025-04-14"
    OPENAI_COMP_MODEL = "gpt-3.5-turbo-instruct"
    OPENAI_BASE_URL = CHATGPT_BASE_URL

else:
    logger.error(f'Invalid DEFAULT_MODEL: {DEFAULT_MODEL}')
    exit(1)


# Don't touch this!
os.environ['OPENAI_BASE_URL'] = OPENAI_BASE_URL
if OPENAI_API_KEY is not None:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
