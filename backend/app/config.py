"""\nConfiguration management\nLoads configuration uniformly from the .env file in the project root\n"""

import os
from dotenv import load_dotenv

from .llm_provider import (
    PROVIDER_ANTHROPIC,
    SUPPORTED_EFFORTS,
    SUPPORTED_PROVIDERS,
    resolve_provider,
)

# Load the .env file from the project root
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # If there is no .env in the root, try loading environment variables (for production)
    load_dotenv(override=True)


class Config:
    """Flask configuration class"""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # JSON configuration - disable ASCII escaping so non-ASCII text displays directly
    JSON_AS_ASCII = False
    
    # LLM configuration (OpenAI format, or native Claude API)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    LLM_PROVIDER = resolve_provider(os.environ.get('LLM_PROVIDER'), LLM_MODEL_NAME)
    # Model for simulation agents, which make the most calls. Falls back to LLM_MODEL_NAME.
    LLM_SIMULATION_MODEL_NAME = os.environ.get('LLM_SIMULATION_MODEL_NAME') or LLM_MODEL_NAME
    LLM_BASE_URL = os.environ.get(
        'LLM_BASE_URL',
        None if LLM_PROVIDER == PROVIDER_ANTHROPIC else 'https://api.openai.com/v1',
    )
    # Claude only: low, medium, high, xhigh or max. Unset uses the model's default.
    LLM_EFFORT = (os.environ.get('LLM_EFFORT') or '').strip().lower() or None
    
    # Knowledge graph: self-hosted Graphiti with FalkorDB
    FALKORDB_HOST = os.environ.get('FALKORDB_HOST', 'localhost')
    FALKORDB_PORT = int(os.environ.get('FALKORDB_PORT', '6379'))
    FALKORDB_USERNAME = os.environ.get('FALKORDB_USERNAME') or None
    FALKORDB_PASSWORD = os.environ.get('FALKORDB_PASSWORD') or None
    GRAPH_STORE_URL = f'falkor://{FALKORDB_HOST}:{FALKORDB_PORT}'
    # Graph extraction makes many calls, so it defaults to a low-cost model.
    GRAPH_LLM_MODEL_NAME = os.environ.get('GRAPH_LLM_MODEL_NAME') or (
        'claude-haiku-5-5' if LLM_PROVIDER == PROVIDER_ANTHROPIC else LLM_MODEL_NAME
    )
    GRAPH_EMBEDDING_MODEL = os.environ.get(
        'GRAPH_EMBEDDING_MODEL',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    )
    GRAPH_EMBEDDING_DIM = int(os.environ.get('GRAPH_EMBEDDING_DIM', '384'))
    GRAPH_MAX_CONCURRENCY = int(os.environ.get('GRAPH_MAX_CONCURRENCY', '8'))
    
    # File upload configuration
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # Text processing configuration
    DEFAULT_CHUNK_SIZE = 500  # Default chunk size
    DEFAULT_CHUNK_OVERLAP = 50  # Default overlap size
    
    # OASIS simulation configuration
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS platform available actions configuration
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent configuration
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration"""
        errors: list[str] = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is not configured")
        if cls.LLM_PROVIDER not in SUPPORTED_PROVIDERS:
            errors.append(f"LLM_PROVIDER is not supported: {cls.LLM_PROVIDER}")
        if cls.LLM_EFFORT and cls.LLM_EFFORT not in SUPPORTED_EFFORTS:
            errors.append(f"LLM_EFFORT is not supported: {cls.LLM_EFFORT}")
        if (
            cls.LLM_PROVIDER == PROVIDER_ANTHROPIC
            and 'openai.com' in (cls.LLM_BASE_URL or '')
        ):
            errors.append("LLM_BASE_URL points to OpenAI, but LLM_PROVIDER is anthropic")
        if (
            resolve_provider(None, cls.LLM_SIMULATION_MODEL_NAME)
            != resolve_provider(None, cls.LLM_MODEL_NAME)
        ):
            errors.append("LLM_SIMULATION_MODEL_NAME must use the same provider as LLM_MODEL_NAME")
        if cls.DEBUG:
            import warnings
            warnings.warn("Flask DEBUG mode is enabled. Do not use in production.", RuntimeWarning)
        return errors
