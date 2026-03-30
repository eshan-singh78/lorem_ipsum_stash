"""Unit tests for docker-compose.yml configuration.

Validates: Requirements 1.1, 1.2, 1.4, 1.5, 1.6
"""
import yaml
from pathlib import Path

COMPOSE_PATH = Path(__file__).parent.parent / "docker-compose.yml"


def load_compose():
    with COMPOSE_PATH.open() as f:
        return yaml.safe_load(f)


def test_all_services_present():
    """All 6 expected services are defined."""
    compose = load_compose()
    services = set(compose["services"].keys())
    expected = {"postgres", "ollama", "ollama-pull", "r2r", "r2r-dashboard", "sebi-api"}
    assert expected == services


def test_port_mappings():
    """Each service exposes the correct host:container port mapping."""
    compose = load_compose()
    services = compose["services"]

    expected_ports = {
        "postgres": "5432:5432",
        "ollama": "11434:11434",
        "r2r": "7272:7272",
        "r2r-dashboard": "3000:3000",
        "sebi-api": "8000:8000",
    }

    for service, port in expected_ports.items():
        assert port in services[service]["ports"], (
            f"{service} missing port mapping {port}"
        )


def test_dependency_conditions():
    """Services declare the correct depends_on conditions."""
    compose = load_compose()
    services = compose["services"]

    def dep_condition(service, dependency):
        return services[service]["depends_on"][dependency]["condition"]

    assert dep_condition("r2r", "postgres") == "service_healthy"
    assert dep_condition("r2r", "ollama") == "service_healthy"
    assert dep_condition("r2r-dashboard", "r2r") == "service_healthy"
    assert dep_condition("sebi-api", "r2r") == "service_healthy"
    assert dep_condition("ollama-pull", "ollama") == "service_healthy"


def test_named_volumes():
    """Top-level volumes section contains both named volumes."""
    compose = load_compose()
    volumes = compose.get("volumes", {})

    # Collect the `name:` values from each volume entry
    named = {v["name"] for v in volumes.values() if isinstance(v, dict) and "name" in v}

    assert "sebi_postgres_data" in named
    assert "sebi_ollama_data" in named


def test_ollama_pull_restart_policy():
    """ollama-pull service has restart policy set to 'no'."""
    compose = load_compose()
    restart = compose["services"]["ollama-pull"]["restart"]
    assert restart == "no"
