"""LLM provider configuration helpers for the Streamlit app."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from conversation.client import (
    get_api_key_for_provider,
    get_gemini_location,
    get_gemini_mode,
    get_gemini_project,
    has_gemini_adc_credentials,
    resolve_model,
    resolve_provider,
)


def get_llm_provider() -> str:
    try:
        if "LLM_PROVIDER" in st.secrets:
            return resolve_provider(st.secrets["LLM_PROVIDER"])
    except Exception:
        pass
    return resolve_provider(os.environ.get("LLM_PROVIDER"))


def get_provider_api_key(provider: str) -> str | None:
    secret_name = "ANTHROPIC_API_KEY" if provider == "anthropic" else "GEMINI_API_KEY"
    try:
        if secret_name in st.secrets:
            return st.secrets[secret_name]
    except Exception:
        pass
    return get_api_key_for_provider(provider)


def get_provider_model(provider: str) -> str:
    secret_name = "ANTHROPIC_MODEL" if provider == "anthropic" else "GEMINI_MODEL"
    try:
        if secret_name in st.secrets:
            return resolve_model(provider, st.secrets[secret_name])
    except Exception:
        pass
    return resolve_model(provider, os.environ.get(secret_name))


def provider_label(provider: str) -> str:
    return "Anthropic" if provider == "anthropic" else "Google Gemini"


def get_gcp_service_account_info() -> dict[str, Any] | None:
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
    except Exception:
        pass
    return None


def get_gemini_vertex_credentials():
    info = get_gcp_service_account_info()
    if not info:
        return None
    try:
        from google.oauth2 import service_account
        return service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    except Exception:
        return None


def gemini_status() -> tuple[bool, str]:
    mode = get_gemini_mode()
    if mode == "vertexai":
        project = get_gemini_project()
        location = get_gemini_location()
        if not project or not location:
            return False, "Vertex AI config missing project or location"
        service_account_info = get_gcp_service_account_info()
        if service_account_info:
            if get_gemini_vertex_credentials() is not None:
                return True, f"Vertex service account ready · {project} / {location}"
            return False, f"Vertex service account secret invalid · {project} / {location}"
        if has_gemini_adc_credentials():
            return True, f"Vertex AI ADC ready · {project} / {location}"
        return False, f"Vertex AI configured but ADC not detected · {project} / {location}"

    if get_provider_api_key("gemini"):
        return True, "Gemini Developer API key ready"
    return False, "Gemini Developer API key not configured"
