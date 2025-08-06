"""
Tests for configuration module.
"""

import pytest
from src.config.settings import Settings, Environment


def test_settings_initialization():
    """Test that settings can be initialized."""
    settings = Settings()
    assert settings.environment in [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]


def test_development_settings():
    """Test development environment settings."""
    settings = Settings(Environment.DEVELOPMENT.value)
    assert settings.environment == Environment.DEVELOPMENT
    assert settings.api.debug is True
    assert settings.api.reload is True


def test_production_settings():
    """Test production environment settings."""
    settings = Settings(Environment.PRODUCTION.value)
    assert settings.environment == Environment.PRODUCTION
    assert settings.api.debug is False
    assert settings.api.reload is False
    assert settings.model.max_workers == 10 