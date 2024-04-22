import re

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    BaseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import AgentAnswer, AnswerEvaluatorTypes, Query
from ragelo.types.configurations import PairwiseEvaluatorConfig


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.PAIRWISE)
class PairwiseEvaluator(BaseAnswerEvaluator):
    """A evaluator that evaluates RAG-based answers pairwise, with document reasoning"""

    config: PairwiseEvaluatorConfig
    output_columns: list[str] = ["qid", "agent_a", "agent_b", "raw_answer", "answer"]
    document_template: str = (
        "[RETRIEVED DOCUMENT]\n{doc}\n[DOCUMENT RELEVANCE]\n{annotation}\n"
    )
    output_file: str = "pairwise_answers_evaluations.csv"
    prompt = """
Please act as an impartial judge and evaluate the quality of the responses provided \
by two AI assistants tasked to answer the question displayed below, based on a set \
of documents retrieved by a search engine.
You should choose the assistant that best answers the user question based on a set \
of reference documents that may or not be relevant.
Will you be provided with the text of each reference document, as well as a reasoning \
why the document is or is not relevant.
Your evaluation should consider factors such as the comprehensiveness, \
correctness, helpfulness, completeness, accuracy, depth, and level of detail \
of their responses. Answers are comprehensive if they show the user multiple \
perspectives in addition to but still relevant to the intent of the original \
question. 
Begin your evaluation by explaining why each answer correctly answers the user \
question. Then, you should compare the two responses and provide a short explanation \
on their differences. Avoid any position biases and ensure that the order in which \
the responses were presented does not influence your decision. Do not allow the \
length of the responses to influence your evaluation. Be as objective as possible.
After providing your explanation, output your final verdict by strictly following \
this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, \
and "[[C]]" for a tie.

[User Question]
{query}

[Reference Documents]
{documents}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
""".strip()

    def __init__(
        self,
        config: PairwiseEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        self.pattern = re.compile(r"\[\[([^]]+)]].*$(?:(?!\[\[).)*", re.DOTALL)

    def _build_message_pairwise(
        self, query: Query, answer: AgentAnswer | tuple[AgentAnswer, AgentAnswer]
    ) -> str:
        assert isinstance(answer, tuple)
        documents = self._prepare_documents(query)
        query_metadata = self._get_usable_fields_from_metadata(
            self.prompt, query.metadata, skip_fields=[self.config.query_placeholder]
        )
        answer_a_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
            answer[0].metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        answer_b_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
            answer[1].metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        formatters = {
            self.config.query_placeholder: query.query,
            self.config.documents_placeholder: documents,
            "answer_a": answer[0].text,
            "answer_b": answer[1].text,
            **query_metadata,
            **answer_a_metadata,
            **answer_b_metadata,
        }
        return self.prompt.format(**formatters)

    def _process_answer(self, answer: str) -> str:
        """Extracts the relevant part of an answer."""
        match_ans = self.pattern.search(answer)
        if not match_ans:
            raise ValueError(f"Could not find answer in {answer}")
        answer = match_ans.group(1)
        if answer not in ["A", "B", "C"]:
            raise ValueError(f"Unknown answer: {answer}")
        return answer
