import json
from unittest.mock import Mock, patch

import pytest
from openai import OpenAI
from openai.resources.chat import Chat
from openai.resources.chat.completions import Completions

from ragelo.llm_providers.base_llm_provider import (
    BaseLLMProvider,
    LLMProviderConfiguration,
)
from ragelo.llm_providers.openai_client import OpenAIConfiguration
from ragelo.types.configurations import RetrievalEvaluatorConfig


@pytest.fixture
def openai_client_config():
    return OpenAIConfiguration(
        api_key="fake key",
        openai_org="fake org",
        openai_api_type="open_ai",
        openai_api_base=None,
        openai_api_version=None,
    )


@pytest.fixture
def llm_provider_config():
    return LLMProviderConfiguration(
        api_key="fake key",
    )


@pytest.fixture
def chat_completion_mock(mocker):
    return mocker.Mock(Completions)


@pytest.fixture
def openai_client_mock(mocker, chat_completion_mock):
    openai_client = mocker.Mock(OpenAI)
    type(openai_client).chat = mocker.Mock(Chat)
    type(openai_client.chat).completions = mocker.PropertyMock(
        return_value=chat_completion_mock
    )
    return openai_client


@pytest.fixture
def retrieval_eval_config():
    return RetrievalEvaluatorConfig(
        documents_path="tests/data/documents.csv",
        query_path="tests/data/queries.csv",
        output_file="tests/data/output.csv",
        force=True,
        verbose=True,
    )


@pytest.fixture
def rdnam_config():
    return RetrievalEvaluatorConfig(
        documents_path="tests/data/documents.csv",
        query_path="tests/data/queries.csv",
        output_file="tests/data/output.csv",
        force=True,
        role="You are a search quality rater evaluating the relevance of web pages. ",
        aspects=True,
        multiple=True,
        narrative_file="tests/data/rdnam_narratives.csv",
        description_file="tests/data/rdnam_descriptions.csv",
    )


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, config):
        self.config = config

    @classmethod
    def from_configuration(cls, config: LLMProviderConfiguration):
        return cls(config)

    def inner_call(self, prompt) -> str:
        return f"Processed {prompt}"

    def __call__(self, prompt) -> str:
        return self.inner_call(prompt)


# @patch.object(MockLLMProvider, "__call__", autospec=True)
@pytest.fixture
def llm_provider_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    provider.inner_call = Mock(side_effect=lambda prompt: f"Processed {prompt}")
    return provider


@pytest.fixture
def llm_provider_mock_rdnam(llm_provider_config):
    mocked_scores = [{"M": 2, "T": 1, "O": 1}, {"M": 1, "T": 1, "O": 2}]
    provider = MockLLMProvider(llm_provider_config)
    provider.inner_call = Mock(side_effect=lambda _: json.dumps(mocked_scores)[2:])
    return provider


# @pytest.fixture
# def base_retrieval_evaluator_mock(retrieval_eval_config, LLM_provider_mock):


# @pytest.fixture
# def temp_file():
#     with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
#         yield f.name
#     os.unlink(f.name)


# @pytest.fixture
# def mock_logger():
#     with patch("ragelo.doc_evaluators.base_retrieval_evaluator.logger") as mock:
#         yield mock


# @pytest.fixture
# def mock_progress_bar():
#     with patch("ragelo.doc_evaluators.base_retrieval_evaluator.Progress") as mock:
#         yield mock


# @pytest.fixture
# def reasoner_evaluator():
#     from ragelo.retrieval_evaluators.reasoner_evaluator import ReasonerEvaluator

#     return ReasonerEvaluator


# @pytest.fixture
# def mock_retrieval_evaluator_config(temp_file):
#     config = RetrievalEvaluatorConfig(
#         documents_path="tests/data/documents.csv",
#         query_path="tests/data/queries.csv",
#         output_file=temp_file,
#     )
#     return config


# @pytest.fixture
# def document_evaluator(mock_openai_client, mock_retrieval_evaluator_config):
#     evaluator = MockRetrievalEvaluator.create_from_config(
#         config=mock_retrieval_evaluator_config,
#         llm_provider=mock_openai_client,
#     )
#     return evaluator


# @pytest.fixture
# def mock_openai_client_call():
#     with patch(
#         "ragelo.llm_providers.openai_client.OpenAIModel.__call__", autospec=True
#     ) as mock:
#         yield mock


# @pytest.fixture
# def mock_process_answer():
#     with patch(
#         "ragelo.doc_evaluators.base_retrieval_evaluator.RetrievalEvaluator._process_answer",
#         autospec=True,
#     ) as mock:
#         yield mock
