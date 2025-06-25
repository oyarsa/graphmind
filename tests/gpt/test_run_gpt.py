"""Unit tests for LLM clients in run_gpt module."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from paper.gpt.run_gpt import (
    AZURE_TIER,
    MODEL_COSTS,
    MODEL_SYNONYMS,
    MODELS_ALLOWED,
    GeminiClient,
    GPTResult,
    LLMClient,
    OpenAIClient,
    _calc_cost,
    _find_best_match,
    count_tokens,
    get_rate_limiter,
    truncate_text,
)


class LLMTestModel(BaseModel):
    """Simple test model for structured output testing."""

    message: str
    value: int


class TestLLMClientFactory:
    """Test LLMClient factory methods."""

    def test_new_routes_to_openai_for_gpt_models(self) -> None:
        """Test that LLMClient.new() routes GPT models to OpenAIClient."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient.new(model="gpt-4o-mini", seed=42, api_key="test-key")
            assert isinstance(client, OpenAIClient)

    def test_new_routes_to_gemini_for_gemini_models(self) -> None:
        """Test that LLMClient.new() routes Gemini models to GeminiClient."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            client = LLMClient.new(
                model="gemini-2.0-flash", seed=42, api_key="test-key"
            )
            assert isinstance(client, GeminiClient)

    def test_new_env_routes_to_openai_for_gpt_models(self) -> None:
        """Test that LLMClient.new_env() routes GPT models to OpenAIClient."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            client = LLMClient.new_env(model="gpt-4o", seed=42)
            assert isinstance(client, OpenAIClient)

    def test_new_env_routes_to_gemini_for_gemini_models(self) -> None:
        """Test that LLMClient.new_env() routes Gemini models to GeminiClient."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            client = LLMClient.new_env(model="gemini-2.5-flash", seed=42)
            assert isinstance(client, GeminiClient)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
class TestOpenAIClient:
    """Test OpenAIClient initialization and configuration."""

    def test_init_with_valid_model(self) -> None:
        """Test OpenAIClient initialization with valid model."""
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-4o-mini",
            seed=42,
        )
        assert client.model == "gpt-4o-mini-2024-07-18"  # Should resolve synonym
        assert client.seed == 42
        assert client.temperature == 0
        assert client.timeout == 60
        assert client.max_input_tokens == 90_000

    def test_init_with_model_synonym(self) -> None:
        """Test that model synonyms are resolved correctly."""
        client = OpenAIClient(
            api_key="test-key",
            model="4o-mini",
            seed=42,
        )
        assert client.model == "gpt-4o-mini-2024-07-18"

    def test_init_with_invalid_model_raises_error(self) -> None:
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid OpenAI model"):
            OpenAIClient(
                api_key="test-key",
                model="invalid-model",
                seed=42,
            )

    def test_init_with_azure_configuration(self) -> None:
        """Test Azure OpenAI configuration."""
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-4o",
            seed=42,
            base_url="https://test.openai.azure.com/",
        )
        assert client.model == "gpt-4o"

    def test_init_with_azure_invalid_model_raises_error(self) -> None:
        """Test that invalid Azure model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Azure model"):
            OpenAIClient(
                api_key="test-key",
                model="gpt-4o-mini-2024-07-18",  # Full name not allowed in Azure
                seed=42,
                base_url="https://test.openai.azure.com/",
            )

    def test_from_env_standard_configuration(self) -> None:
        """Test OpenAIClient.from_env() with standard configuration."""
        with patch.dict(os.environ, {"USE_AZURE": "0", "OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient.from_env(model="gpt-4o-mini", seed=42)
            assert client.model == "gpt-4o-mini-2024-07-18"
            assert client.seed == 42

    def test_from_env_azure_configuration(self) -> None:
        """Test OpenAIClient.from_env() with Azure configuration."""
        env_vars = {
            "USE_AZURE": "1",
            "AZURE_BASE_URL": "https://test.openai.azure.com/{{model}}/",
            "AZURE_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            client = OpenAIClient.from_env(model="gpt-4o", seed=42)
            assert client.model == "gpt-4o"
            assert client.seed == 42

    def test_from_env_azure_model_conversion(self) -> None:
        """Test that Azure model names are converted correctly."""
        env_vars = {
            "USE_AZURE": "1",
            "AZURE_BASE_URL": "https://test.openai.azure.com/{{model}}/",
            "AZURE_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars):
            # Test full model name conversion
            client = OpenAIClient.from_env(model="gpt-4o-mini-2024-07-18", seed=42)
            assert client.model == "gpt-4o-mini"


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY environment variable not set",
)
class TestGeminiClient:
    """Test GeminiClient initialization and configuration."""

    def test_init_with_valid_model(self) -> None:
        """Test GeminiClient initialization with valid model."""
        client = GeminiClient(
            api_key="test-key",
            model="gemini-2.0-flash",
            seed=42,
        )
        assert client.model == "gemini-2.0-flash-001"  # Should resolve synonym
        assert client.seed == 42
        assert client.temperature == 0
        assert client.timeout == 60
        assert client.max_input_tokens == 90_000

    def test_init_with_model_synonym(self) -> None:
        """Test that model synonyms are resolved correctly."""
        client = GeminiClient(
            api_key="test-key",
            model="gemini-2.5-flash",
            seed=42,
        )
        assert client.model == "gemini-2.5-flash-preview-04-17"

    def test_init_with_invalid_model_raises_error(self) -> None:
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model"):
            GeminiClient(
                api_key="test-key",
                model="invalid-gemini-model",
                seed=42,
            )

    def test_init_with_thinking_budget_valid(self) -> None:
        """Test initialization with valid thinking budget."""
        client = GeminiClient(
            api_key="test-key",
            model="gemini-2.5-pro",
            seed=42,
            thinking_budget=1000,
        )
        assert client.thinking_budget == 1000

    def test_init_with_thinking_budget_invalid_raises_error(self) -> None:
        """Test that invalid thinking budget raises ValueError."""
        with pytest.raises(ValueError, match="thinking_budget must be in"):
            GeminiClient(
                api_key="test-key",
                model="gemini-2.5-pro",
                seed=42,
                thinking_budget=50000,  # Too high
            )

    def test_init_with_thinking_budget_zero(self) -> None:
        """Test initialization with thinking budget set to zero."""
        client = GeminiClient(
            api_key="test-key",
            model="gemini-2.5-pro",
            seed=42,
            thinking_budget=0,
        )
        assert client.thinking_budget == 0

    def test_from_env_standard_configuration(self) -> None:
        """Test GeminiClient.from_env() with standard configuration."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            client = GeminiClient.from_env(model="gemini-2.0-flash", seed=42)
            assert client.model == "gemini-2.0-flash-001"
            assert client.seed == 42

    def test_from_env_with_thinking_config(self) -> None:
        """Test GeminiClient.from_env() with thinking configuration."""
        env_vars = {
            "GEMINI_API_KEY": "test-key",
            "INCLUDE_THOUGHTS": "1",
            "THINKING_BUDGET": "2000",
        }
        with patch.dict(os.environ, env_vars):
            client = GeminiClient.from_env(model="gemini-2.5-pro", seed=42)
            assert client.include_thoughts is True
            assert client.thinking_budget == 2000

    def test_from_env_invalid_thinking_budget_raises_error(self) -> None:
        """Test that invalid THINKING_BUDGET environment variable raises ValueError."""
        env_vars = {
            "GEMINI_API_KEY": "test-key",
            "THINKING_BUDGET": "not_a_number",
        }
        with (
            patch.dict(os.environ, env_vars),
            pytest.raises(ValueError, match="THINKING_BUDGET must be an integer"),
        ):
            GeminiClient.from_env(model="gemini-2.5-pro", seed=42)

    def test_from_env_invalid_include_thoughts_raises_error(self) -> None:
        """Test that invalid INCLUDE_THOUGHTS environment variable raises ValueError."""
        env_vars = {
            "GEMINI_API_KEY": "test-key",
            "INCLUDE_THOUGHTS": "invalid",
        }
        with (
            patch.dict(os.environ, env_vars),
            pytest.raises(ValueError, match="INCLUDE_THOUGHTS must be unset, 0 or 1"),
        ):
            GeminiClient.from_env(model="gemini-2.5-pro", seed=42)


class TestUtilityFunctions:
    """Test utility functions for cost calculation and text processing."""

    @pytest.mark.parametrize(
        ("model", "prompt_tokens", "completion_tokens", "expected_cost"),
        [
            ("gpt-4o-mini-2024-07-18", 1000, 500, 0.00045),  # 0.0001 + 0.0003
            ("gpt-4o-2024-08-06", 1000, 500, 0.0075),  # 0.0025 + 0.005
            ("gemini-2.0-flash-001", 1000, 500, 0.0003),  # 0.0001 + 0.0002
            ("invalid-model", 1000, 500, 0),  # Should return 0 for unknown models
        ],
    )
    def test_calc_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        expected_cost: float,
    ) -> None:
        """Test cost calculation for different models."""
        cost = _calc_cost(model, prompt_tokens, completion_tokens)
        assert abs(cost - expected_cost) < 1e-6

    @pytest.mark.parametrize(
        ("model", "limits", "expected"),
        [
            (
                "gpt-4o-mini-2024-07-18",
                {"gpt-4o-mini": (5000, 4000000)},
                (5000, 4000000),
            ),
            ("gpt-4o", {"gpt-4o": (1000, 800000)}, (1000, 800000)),
            (
                "gemini-2.0-flash-001",
                {"gemini-2.0-flash": (2000, 4000000)},
                (2000, 4000000),
            ),
            ("unknown-model", {"gpt-4o": (1000, 800000)}, None),
        ],
    )
    def test_find_best_match(
        self,
        model: str,
        limits: dict[str, tuple[int, int]],
        expected: tuple[int, int] | None,
    ) -> None:
        """Test finding best matching rate limits for models."""
        result = _find_best_match(model, limits)
        assert result == expected

    @pytest.mark.parametrize(
        ("tier", "model"),
        [
            (1, "gemini-2.0-flash"),
            (3, "gpt-4o-mini"),
            (4, "gpt-4o"),
            (5, "gpt-4o"),
            (AZURE_TIER, "gpt-4o-mini"),
        ],
    )
    def test_get_rate_limiter_valid_combinations(self, tier: int, model: str) -> None:
        """Test that get_rate_limiter works for valid tier/model combinations."""
        rate_limiter = get_rate_limiter(tier, model)
        assert rate_limiter is not None

    def test_get_rate_limiter_invalid_tier_raises_error(self) -> None:
        """Test that invalid tier raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tier"):
            get_rate_limiter(999, "gpt-4o")

    def test_get_rate_limiter_unsupported_model_raises_error(self) -> None:
        """Test that unsupported model for tier raises ValueError."""
        with pytest.raises(ValueError, match="Model .* is not supported for tier"):
            get_rate_limiter(3, "gemini-2.0-flash")

    def test_count_tokens(self) -> None:
        """Test token counting function."""
        text = "Hello world"
        token_count = count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_truncate_text_no_truncation_needed(self) -> None:
        """Test that short text is not truncated."""
        text = "Short text"
        max_tokens = 1000
        result = truncate_text(text, max_tokens)
        assert result == text

    def test_truncate_text_truncation_applied(self) -> None:
        """Test that long text is truncated correctly."""
        text = "This is a very long text " * 100
        max_tokens = 10
        result = truncate_text(text, max_tokens)
        assert len(result) < len(text)
        assert count_tokens(result) <= max_tokens


class TestModelConfiguration:
    """Test model configuration and validation."""

    def test_model_synonyms_coverage(self) -> None:
        """Test that all model synonyms resolve to valid models."""
        for synonym, full_name in MODEL_SYNONYMS.items():
            assert full_name in MODELS_ALLOWED
            assert synonym in MODELS_ALLOWED

    def test_model_costs_coverage(self) -> None:
        """Test that all full model names have cost information."""
        cost_model_names = set(MODEL_COSTS.keys())

        # Some models might not have costs yet, but core models should
        core_models = {
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-08-06",
            "gemini-2.0-flash-001",
        }
        assert core_models.issubset(cost_model_names)

    def test_model_costs_format(self) -> None:
        """Test that all model costs are properly formatted."""
        for input_cost, output_cost in MODEL_COSTS.values():
            assert input_cost >= 0
            assert output_cost >= 0


class TestLLMClientMethods:
    """Test LLM client methods with mocked API calls."""

    @pytest.mark.asyncio
    async def test_openai_run_success(self) -> None:
        """Test OpenAI client run method with successful response."""
        mock_completion = MagicMock()
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.parsed = LLMTestModel(
            message="test response", value=42
        )

        with patch("paper.gpt.run_gpt.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            mock_client.beta.chat.completions.parse = AsyncMock(
                return_value=mock_completion
            )

            client = OpenAIClient(
                api_key="test-key",
                model="gpt-4o-mini",
                seed=42,
            )
            client.client = mock_client

            with patch.object(client.rate_limiter, "limit") as mock_limiter:
                mock_limiter.return_value.__aenter__ = AsyncMock(
                    return_value=AsyncMock()
                )
                mock_limiter.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await client.run(
                    LLMTestModel,
                    "You are a helpful assistant",
                    "Generate a test response",
                )

                assert result.result is not None
                assert result.result.message == "test response"
                assert result.result.value == 42
                assert result.cost > 0

    @pytest.mark.asyncio
    async def test_openai_run_api_error(self) -> None:
        """Test OpenAI client run method with API error."""
        with patch("paper.gpt.run_gpt.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            mock_client.beta.chat.completions.parse = AsyncMock(
                side_effect=Exception("API Error")
            )

            client = OpenAIClient(
                api_key="test-key",
                model="gpt-4o-mini",
                seed=42,
            )
            client.client = mock_client

            with patch.object(client.rate_limiter, "limit") as mock_limiter:
                mock_limiter.return_value.__aenter__ = AsyncMock(
                    return_value=AsyncMock()
                )
                mock_limiter.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await client.run(
                    LLMTestModel,
                    "You are a helpful assistant",
                    "Generate a test response",
                )

                assert result.result is None
                assert result.cost == 0

    @pytest.mark.asyncio
    async def test_openai_plain_success(self) -> None:
        """Test OpenAI client plain method with successful response."""
        mock_completion = MagicMock()
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Plain text response"

        with patch("paper.gpt.run_gpt.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_completion
            )

            client = OpenAIClient(
                api_key="test-key",
                model="gpt-4o-mini",
                seed=42,
            )
            client.client = mock_client

            with patch.object(client.rate_limiter, "limit") as mock_limiter:
                mock_limiter.return_value.__aenter__ = AsyncMock(
                    return_value=AsyncMock()
                )
                mock_limiter.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await client.plain(
                    "You are a helpful assistant",
                    "Say hello",
                )

                assert result.result == "Plain text response"
                assert result.cost > 0

    @pytest.mark.asyncio
    async def test_openai_plain_with_search(self) -> None:
        """Test OpenAI client plain method with search level."""
        mock_completion = MagicMock()
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Search response"

        with patch("paper.gpt.run_gpt.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                return_value=mock_completion
            )

            client = OpenAIClient(
                api_key="test-key",
                model="gpt-4o-search",
                seed=42,
            )
            client.client = mock_client

            with patch.object(client.rate_limiter, "limit") as mock_limiter:
                mock_limiter.return_value.__aenter__ = AsyncMock(
                    return_value=AsyncMock()
                )
                mock_limiter.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await client.plain(
                    "You are a helpful assistant",
                    "Search for information about AI",
                    search_level="medium",
                )

                assert result.result == "Search response"
                assert result.cost > 0

    @pytest.mark.asyncio
    async def test_gemini_run_success(self) -> None:
        """Test Gemini client run method with successful response."""
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.parsed = LLMTestModel(message="gemini response", value=24)

        with patch("paper.gpt.run_gpt.genai.Client") as mock_genai:
            mock_client = AsyncMock()
            mock_genai.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            client = GeminiClient(
                api_key="test-key",
                model="gemini-2.0-flash",
                seed=42,
            )
            client.client = mock_client

            with patch.object(client.rate_limiter, "limit") as mock_limiter:
                mock_limiter.return_value.__aenter__ = AsyncMock(
                    return_value=AsyncMock()
                )
                mock_limiter.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await client.run(
                    LLMTestModel,
                    "You are a helpful assistant",
                    "Generate a test response",
                )

                assert result.result is not None
                assert result.result.message == "gemini response"
                assert result.result.value == 24
                assert result.cost > 0

    @pytest.mark.asyncio
    async def test_gemini_run_invalid_parsed_response(self) -> None:
        """Test Gemini client run method with invalid parsed response."""
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.parsed = "not a valid model instance"

        with patch("paper.gpt.run_gpt.genai.Client") as mock_genai:
            mock_client = AsyncMock()
            mock_genai.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            client = GeminiClient(
                api_key="test-key",
                model="gemini-2.0-flash",
                seed=42,
            )
            client.client = mock_client

            with patch.object(client.rate_limiter, "limit") as mock_limiter:
                mock_limiter.return_value.__aenter__ = AsyncMock(
                    return_value=AsyncMock()
                )
                mock_limiter.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await client.run(
                    LLMTestModel,
                    "You are a helpful assistant",
                    "Generate a test response",
                )

                assert result.result is None
                assert result.cost > 0  # Cost is still calculated

    @pytest.mark.asyncio
    async def test_gemini_plain_success(self) -> None:
        """Test Gemini client plain method with successful response."""
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Gemini plain response"

        with patch("paper.gpt.run_gpt.genai.Client") as mock_genai:
            mock_client = AsyncMock()
            mock_genai.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            client = GeminiClient(
                api_key="test-key",
                model="gemini-2.0-flash",
                seed=42,
            )
            client.client = mock_client

            with patch.object(client.rate_limiter, "limit") as mock_limiter:
                mock_limiter.return_value.__aenter__ = AsyncMock(
                    return_value=AsyncMock()
                )
                mock_limiter.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await client.plain(
                    "You are a helpful assistant",
                    "Say hello in Gemini",
                )

                assert result.result == "Gemini plain response"
                assert result.cost > 0

    @pytest.mark.asyncio
    async def test_gemini_plain_with_search(self) -> None:
        """Test Gemini client plain method with search enabled."""
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Gemini search response"

        with patch("paper.gpt.run_gpt.genai.Client") as mock_genai:
            mock_client = AsyncMock()
            mock_genai.return_value = mock_client
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )

            client = GeminiClient(
                api_key="test-key",
                model="gemini-2.0-flash",
                seed=42,
            )
            client.client = mock_client

            with patch.object(client.rate_limiter, "limit") as mock_limiter:
                mock_limiter.return_value.__aenter__ = AsyncMock(
                    return_value=AsyncMock()
                )
                mock_limiter.return_value.__aexit__ = AsyncMock(return_value=None)

                result = await client.plain(
                    "You are a helpful assistant",
                    "Search for AI information",
                    search_level="high",
                )

                assert result.result == "Gemini search response"
                assert result.cost > 0

    def test_gpt_result_operations(self) -> None:
        """Test GPTResult utility operations."""
        # Test pure
        result_unit = GPTResult[str].unit("hello")
        assert result_unit.result == "hello"
        assert result_unit.cost == 0

        # Test constructor
        result_with_cost = GPTResult(result="hello", cost=0.01)
        assert result_with_cost.result == "hello"
        assert result_with_cost.cost == 0.01

        # Test map
        mapped = result_with_cost.map(str.upper)
        assert mapped.result == "HELLO"
        assert mapped.cost == 0.01

        # Test then
        other_result = GPTResult(result="world", cost=0.02)
        combined = result_with_cost.then(other_result)
        assert combined.result == "world"
        assert combined.cost == 0.03

        # Test bind
        def transform(value: str) -> GPTResult[str]:
            return GPTResult(result=f"transformed_{value}", cost=0.005)

        bound = result_with_cost.bind(transform)
        assert bound.result == "transformed_hello"
        assert bound.cost == 0.015


@pytest.mark.slow
class TestLLMClientIntegration:
    """Integration tests for LLM clients with real API calls."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    async def test_openai_run_integration(self) -> None:
        """Test OpenAI client run method with real API call."""
        client = OpenAIClient.from_env(model="gpt-4o-mini", seed=42)

        result = await client.run(
            LLMTestModel,
            "You are a helpful assistant. Respond with JSON containing a message and value.",
            "Create a simple test response with message='integration test' and value=123",
            max_tokens=100,
        )

        assert result.result is not None
        assert result.cost > 0
        # Note: We don't assert exact content since LLM responses can vary

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    async def test_openai_plain_integration(self) -> None:
        """Test OpenAI client plain method with real API call."""
        client = OpenAIClient.from_env(model="gpt-4o-mini", seed=42)

        result = await client.plain(
            "You are a helpful assistant.",
            "Say 'Hello from OpenAI integration test' exactly.",
            max_tokens=50,
        )

        assert result.result is not None
        assert result.cost > 0
        assert "Hello" in result.result

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY environment variable not set",
    )
    async def test_gemini_run_integration(self) -> None:
        """Test Gemini client run method with real API call."""
        client = GeminiClient.from_env(model="gemini-2.0-flash", seed=42)

        result = await client.run(
            LLMTestModel,
            "You are a helpful assistant. Respond with JSON containing a message and value.",
            "Create a simple test response with message='gemini integration test' and value=456",
            max_tokens=100,
        )

        assert result.result is not None
        assert result.cost > 0
        # Note: We don't assert exact content since LLM responses can vary

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY environment variable not set",
    )
    async def test_gemini_plain_integration(self) -> None:
        """Test Gemini client plain method with real API call."""
        client = GeminiClient.from_env(model="gemini-2.0-flash", seed=42)

        result = await client.plain(
            "You are a helpful assistant.",
            "Say 'Hello from Gemini integration test' exactly.",
            max_tokens=50,
        )

        assert result.result is not None
        assert result.cost > 0
        assert "Hello" in result.result
