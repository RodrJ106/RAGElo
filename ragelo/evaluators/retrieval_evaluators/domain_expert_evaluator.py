"""Evaluator with a domain expert persona"""

import logging
from typing import Any, Optional

from tenacity import RetryError

from ragelo.evaluators.retrieval_evaluators import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import Document, Query, RetrievalEvaluatorTypes
from ragelo.types.configurations import DomainExpertEvaluatorConfig


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.DOMAIN_EXPERT)
class DomainExpertEvaluator(BaseRetrievalEvaluator):
    sys_prompt = """
You are a domain expert in {domain_long}.{company_prompt_1} You are tasked \
with evaluating the performance of a retrieval system for question \
answering in this domain. The question answering system will be used \
by internal users{company_prompt_2}{domain_short}. They are interested \
in a retrieval system that provides relevant passages based on their questions.
""".strip()

    reason_prompt = """
User query:
{query}

Document passage:
{doc_content}

Please think in steps about the relevance of the retrieved document given the \
original query. Consider the query, the document title, and the document \
passage. Reason whether the document is not relevant to the query, somewhat relevant \
to the query, or highly relevant to the query.
Use the following guidelines to reason about the relevance of the retrieved document:
- Not Relevant:
    The document contains information that is unrelated, outdated, or completely \
irrelevant to the query.
    The document may contain some keywords or phrases from the query, but the context \
and overall meaning do not align with the query's intent.
    The document may be from a different field or time period, rendering it irrelevant \
to the current query.
- Somewhat Relevant:
    The document contains some relevant information but lacks comprehensive details \
or context.
    The document may discuss a related topic or concept but not directly address the \
query.
    The information in the document is tangentially related to the query, but the \
primary focus remains different.
- Highly Relevant:
    The document directly addresses the main points of the query and provides \
comprehensive and accurate information.
    The document may cite relevant information directly applicable to the query.
    The document may be recent and from the same field as the query, enhancing its \
relevance.
General Guidelines:
    - Context Matters: Annotators should evaluate the relevance of documents within \
the specific{domain_short} context provided by the query. Understanding the nuances \
and domain-specific terminology is essential.
    - Content Overlap: Consider the extent of content overlap between the document\
and the query. Assess whether the document covers the core aspects of the query \
or only peripheral topics.
    - Neutrality: Base judgments solely on the content's relevance and avoid any \
personal opinions or biases.
    - Uncertainty: If uncertain about a relevance judgement, annotators default to a \
lower relevance.
    {extra_guidelines}
""".strip()

    score_prompt = """
Given the previous reasoning, please assign a score of 0, 1, or 2 to the retrieved \
document given the particular query. The score meaning is as follows:
- 0: indicates that the retrieved document is not relevant to the query
- 1: the document is somewhat relevant to the query
- 2: the document is highly relevant to the query
Please only answer with a single number.
""".strip()

    COMPANY_PROMPT_1 = " You work for {company}."
    COMPANY_PROMPT_2 = " of {company}"
    DOMAIN_SHORT = " but it also serves some of your external users like {domain_short}"
    config: DomainExpertEvaluatorConfig
    output_columns: list[str] = ["qid", "did", "reasoning", "score"]
    output_file: str = "domain_expert_evaluations.csv"

    def __init__(
        self,
        config: DomainExpertEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        if not self.config.domain_long:
            raise ValueError(
                "You are tying to use the Domain Expert Retrieval Evaluator. "
                "For this evaluator, you need to provide at least the name of the domain "
                "in the domain_long field."
            )

        self.domain_long = self.config.domain_long
        self.domain_short = (
            f" {self.config.domain_short}" if self.config.domain_short else ""
        )
        self.sys_prompt = self.sys_prompt.format(
            domain_long=self.domain_long,
            company_prompt_1=(
                self.COMPANY_PROMPT_1.format(company=self.config.company)
                if self.config.company
                else ""
            ),
            company_prompt_2=(
                self.COMPANY_PROMPT_2.format(company=self.config.company)
                if self.config.company
                else ""
            ),
            domain_short=(
                self.DOMAIN_SHORT.format(domain_short=self.config.domain_short)
                if self.config.domain_short
                else ""
            ),
        )
        self.extra_guidelines = (
            self.config.extra_guidelines if self.config.extra_guidelines else []
        )

    def __build_reason_message(self, query: Query, document: Document) -> str:
        guidelines = "\n".join(
            [f"- {guideline}" for guideline in self.extra_guidelines]
        )
        reason_prompt = self.reason_prompt.format(
            query=query.query,
            doc_content=document.text,
            domain_short=self.domain_short if self.domain_short else "",
            extra_guidelines=guidelines,
        )
        return reason_prompt

    def evaluate(
        self,
        query: Query | str,
        document: Document | str,
        query_metadata: Optional[dict[str, Any]] = None,
        doc_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Any]:
        """Processes a single pair of qid, did in a two-shot manner"""
        if isinstance(query, str):
            query = Query(qid="<no_qid>", query=query)
        if isinstance(document, str):
            document = Document(did="<no_did>", text=document)
        query.add_metadata(query_metadata)
        document.add_metadata(doc_metadata)

        qid = query.qid
        did = document.did
        reason_message = self.__build_reason_message(query, document)
        messages_reasoning = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": reason_message},
        ]
        try:
            reasoning_answer = self.llm_provider(messages_reasoning)

        except RetryError as e:
            logging.warning(f"Failed to fetch reasoning for document {qid} {did}")
            raise e
        except ValueError as e:
            logging.warning(f"Failed to parse reasoning for document {qid} {did}")
            raise e

        messages_score = messages_reasoning.copy()
        messages_score.append({"role": "assistant", "content": reasoning_answer})
        messages_score.append({"role": "user", "content": self.score_prompt})
        try:
            score_answer = self._process_answer(self.llm_provider(messages_score))
        except RetryError as e:
            logging.warning(f"Failed to fetch evaluation for document {qid} {did}")
            raise e
        except ValueError as e:
            logging.warning(f"Failed to parse evaluation for document {qid} {did}")
            raise e
        return reasoning_answer, score_answer
