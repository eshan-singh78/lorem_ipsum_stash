"""
Unit tests for sebi_ollama.toml R2R configuration.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 14.3
"""

import pathlib

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

TOML_PATH = pathlib.Path(__file__).parent.parent / "config" / "sebi_ollama.toml"


def load_config() -> dict:
    with open(TOML_PATH, "rb") as f:
        return tomllib.load(f)


def test_toml_file_exists():
    assert TOML_PATH.exists(), f"Config file not found: {TOML_PATH}"


def test_app_fast_llm():
    config = load_config()
    assert config["app"]["fast_llm"] == "openai/qwen2.5:3b-instruct"


def test_app_quality_llm():
    config = load_config()
    assert config["app"]["quality_llm"] == "openai/qwen2.5:3b-instruct"


def test_embedding_provider():
    config = load_config()
    assert config["embedding"]["provider"] == "ollama"


def test_embedding_base_model():
    config = load_config()
    assert config["embedding"]["base_model"] == "nomic-embed-text"


def test_embedding_base_dimension():
    config = load_config()
    assert config["embedding"]["base_dimension"] == 768


def test_completion_generation_config_temperature():
    config = load_config()
    assert config["completion"]["generation_config"]["temperature"] == 0.1


def test_completion_generation_config_max_tokens():
    config = load_config()
    assert config["completion"]["generation_config"]["max_tokens_to_sample"] == 512


def test_completion_generation_config_api_base():
    config = load_config()
    assert config["completion"]["generation_config"]["api_base"] == "http://ollama:11434/v1"


def test_ingestion_chunk_size():
    # TOML 3_200 parses to int 3200
    config = load_config()
    assert config["ingestion"]["chunk_size"] == 3200


def test_ingestion_chunk_overlap():
    config = load_config()
    assert config["ingestion"]["chunk_overlap"] == 600


def test_database_vector_search_strategy():
    config = load_config()
    assert config["database"]["vector_search_settings"]["search_strategy"] == "hybrid"
