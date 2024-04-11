"""Base model for dealing with answer evaluators"""

import csv
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Optional, Type, get_type_hints

from tenacity import RetryError
from tqdm import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types import AgentAnswer, AnswerEvaluatorResult, AnswerEvaluatorTypes, Query
from ragelo.types.configurations import BaseAnswerEvaluatorConfig


class BaseAnswerEvaluator(BaseEvaluator):
    config: BaseAnswerEvaluatorConfig
    output_columns = ["qid", "agent", "raw_answer", "answer"]
    output_file: str = "answers_evaluations.csv"
    tuple_columns: list[str] = ["qid", "agent"]

    def __init__(
        self,
        config: BaseAnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider
        if config.output_file is not None:
            self.output_file = config.output_file

    def batch_evaluate(self, queries: list[Query]) -> list[AnswerEvaluatorResult]:
        use_progress_bar = self.config.verbose
        failed_evaluations = 0
        evaluations = [AnswerEvaluatorResult(**x) for x in self._get_existing_output()]
        skip_tuples = {(x.qid, x.agent) for x in evaluations}
        tuples_to_eval = []
        all_tuples = 0
        for query in queries:
            for agent_answer in query.answers:
                qid = query.qid
                agent = agent_answer.agent
                all_tuples += 1
                if (qid, agent) in skip_tuples:
                    logger.debug(f"Skipping {qid} {agent}")
                    continue
                tuples_to_eval.append((query, agent_answer))
        if len(tuples_to_eval) == 0:
            logger.info("All answers have been evaluated")
            if self.config.verbose:
                print(
                    f"All {all_tuples} answers are already evaluated.\n"
                    "If you want to re-evaluate them, use the force flag"
                )
            return evaluations
        for query, agent_answer in tqdm(
            tuples_to_eval,
            desc="Annotating Answers",
            disable=not use_progress_bar,
            ncols=100,
            leave=False,
            position=0,
        ):
            qid = query.qid
            agent = agent_answer.agent
            try:
                raw_answer, answer = self.evaluate(query, agent_answer)
            except (RetryError, ValueError):
                failed_evaluations += 1
                continue
            evaluations.append(
                AnswerEvaluatorResult(
                    qid=qid, agent=agent, raw_answer=raw_answer, answer=answer
                )
            )
            self._dump_response(evaluations[-1], self.output_columns, self.output_file)
        if self.config.verbose:
            print("✅ Done!")
            print(f"Unparsed answers: {failed_evaluations}")
            print(f"Total evaluations: {len(evaluations)}")
        return evaluations

    def evaluate(
        self,
        query: Query | str,
        answer: AgentAnswer | str,
        query_metadata: Optional[dict[str, Any]] = None,
        answer_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Any]:
        if isinstance(query, str):
            query = Query(qid="<no_qid>", query=query)
        if isinstance(answer, str):
            answer = AgentAnswer(agent="<no_agent>", text=answer)
        query.add_metadata(query_metadata)
        answer.add_metadata(answer_metadata)

        message = self._build_message(query, answer)
        try:
            raw_answer = self.llm_provider(message)
        except RetryError as e:
            logger.warning(
                f"Failed to fetch answer for qid: {query.qid} agent: {answer.agent}"
            )
            raise e
        try:
            processed_answer = self._process_answer(raw_answer)

        except ValueError as e:
            logger.warning(
                f"Failed to parse answer for qid: {query.qid} agent: {answer.agent}"
                f"Full answer: {raw_answer}"
            )
            raise e
        return raw_answer, processed_answer

    @abstractmethod
    def _build_message(
        self, query: Query, answer: AgentAnswer
    ) -> str | list[dict[str, str]]:
        """Builds the message to send to the LLM evaluator"""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls, config: BaseAnswerEvaluatorConfig, llm_provider: BaseLLMProvider
    ):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> Type[BaseAnswerEvaluatorConfig]:
        return get_type_hints(cls)["config"]

    @staticmethod
    def _construct_list_of_answers(
        answers: list[dict[str, str]]
    ) -> list[AnswerEvaluatorResult]:
        return [AnswerEvaluatorResult(**x) for x in answers]


class AnswerEvaluatorFactory:
    registry: dict[AnswerEvaluatorTypes | str, Type[BaseAnswerEvaluator]] = {}

    @classmethod
    def register(cls, name: AnswerEvaluatorTypes) -> Callable:
        def inner_wrapper(wrapped_class: Type[BaseAnswerEvaluator]):
            if name in cls.registry:
                logger.warning(f"Overwriting {name} in registry")
            cls.registry[name.lower()] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        evaluator_name: str,
        llm_provider: BaseLLMProvider | str,
        config: Optional[BaseAnswerEvaluatorConfig] = None,
        **kwargs,
    ) -> BaseAnswerEvaluator:
        if evaluator_name.lower() not in cls.registry:
            raise ValueError(f"Unknown evaluator {evaluator_name}")
        if isinstance(llm_provider, str):
            llm_provider_instance = get_llm_provider(llm_provider, **kwargs)
        else:
            llm_provider_instance = llm_provider
        if config is None:
            class_ = cls.registry[evaluator_name]
            type_config = class_.get_config_class()
            valid_keys = [field for field in type_config.__fields__]
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            config = type_config(**valid_args)
        return cls.registry[evaluator_name.lower()].from_config(
            config, llm_provider_instance
        )


def get_answer_evaluator(
    evaluator_name: AnswerEvaluatorTypes | str,
    llm_provider: BaseLLMProvider | str,
    config: Optional[BaseAnswerEvaluatorConfig] = None,
    **kwargs,
) -> BaseAnswerEvaluator:
    return AnswerEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )
