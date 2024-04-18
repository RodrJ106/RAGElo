"""Base model for dealing with answer evaluators"""

import asyncio
import random
from abc import abstractmethod
from string import Formatter
from typing import Any, Callable, Optional, Type, get_type_hints

from aiohttp import ClientSession
from tenacity import RetryError
from tqdm import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types import (
    AgentAnswer,
    AnswerEvaluatorResult,
    AnswerEvaluatorTypes,
    AnswerFormat,
    Document,
    Query,
)
from ragelo.types.configurations import BaseAnswerEvaluatorConfig
from ragelo.types.types import PairwiseGame
from ragelo.utils import load_retrieved_docs_from_csv


class BaseAnswerEvaluator(BaseEvaluator):
    config: BaseAnswerEvaluatorConfig
    output_columns = ["qid", "agent", "raw_answer", "answer"]
    output_file: str = "answers_evaluations.csv"
    document_template: str = "[{did}] {doc}"

    def __init__(
        self,
        config: BaseAnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider
        if config.output_file is not None:
            self.output_file = config.output_file
        if config.answer_format == AnswerFormat.MULTI_FIELD_JSON:
            if isinstance(config.scoring_key, str):
                scoring_keys = [config.scoring_key]
            else:
                scoring_keys = config.scoring_key
            self.output_columns = ["qid", "agent", "raw_answer"] + scoring_keys

        if config.scoring_key and config.scoring_key not in self.output_columns:
            print(f"Adding scoring key {config.scoring_key} to output columns")
            self.output_columns.append(self.config.scoring_key)
        if config.scoring_keys:
            missing_keys = [
                key for key in config.scoring_keys if key not in self.output_columns
            ]
            self.output_columns.extend(missing_keys)

    def __get_tuples_to_evaluate(
        self, queries: list[Query], evaluations: list[AnswerEvaluatorResult]
    ) -> list[tuple[Query, AgentAnswer]]:
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

        return tuples_to_eval

    async def _fetch_chunk_pointwise(
        self, chunk: list[tuple[Query, AgentAnswer]]
    ) -> list[AnswerEvaluatorResult]:
        qids = []
        agent_ids = []
        async with ClientSession() as session:
            tasks = []
            for query, agent_answer in chunk:
                message = self._build_message(query, agent_answer)
                qids.append(query.qid)
                agent_ids.append(agent_answer.agent)
                tasks.append(self.llm_provider.call_async(message, session))
            raw_answers = await asyncio.gather(*tasks)
            parsed_answers = []
            for qid, agent_id, raw_answer in zip(qids, agent_ids, raw_answers):
                try:
                    answer = self._process_answer(raw_answer)
                except ValueError:
                    logger.warning(
                        f"Failed to PARSE answer for qid: {qid} agent: {agent_id}"
                    )
                    continue
                parsed_answers.append(
                    AnswerEvaluatorResult(
                        qid=qid,
                        agent=agent_id,
                        raw_answer=raw_answer,
                        answer=answer,
                    )
                )
                self._dump_response(
                    parsed_answers[-1], self.output_columns, self.output_file
                )
        return parsed_answers

    async def _fetch_chunk_pairwise(
        self, chunk: list[tuple[Query, AgentAnswer]]
    ) -> list[AnswerEvaluatorResult]:
        qids = []
        agent_ids = []
        async with ClientSession() as session:
            tasks = []
            for query, agent_answer_a, agent_answer_b in chunk:
                prompt = self._build_message_pairwise(
                    query, (agent_answer_a, agent_answer_b)
                )
                qids.append(query.qid)
                agent_ids.append((agent_answer_a.agent, agent_answer_b.agent))
                tasks.append(self.llm_provider.call_async(prompt, session))
            raw_answers = await asyncio.gather(*tasks)
            parsed_answers = []
            for qid, agent_id, raw_answer in zip(qids, agent_ids, raw_answers):
                try:
                    answer = self._process_answer(raw_answer)
                except ValueError:
                    logger.warning(
                        f"Failed to PARSE answer for qid: {qid} agent: {agent_id}"
                    )
                    continue
                parsed_answers.append(
                    AnswerEvaluatorResult(
                        qid=qid,
                        agent_a=agent_id[0],
                        agent_b=agent_id[1],
                        pairwise=True,
                        raw_answer=raw_answer,
                        answer=answer,
                    )
                )
                self._dump_response(
                    parsed_answers[-1], self.output_columns, self.output_file
                )
        return parsed_answers

    async def _batch_evaluate_pointwise_async(
        self, queries: list[Query]
    ) -> list[Query]:
        use_progress_bar = self.config.use_progress_bar
        evaluations = [AnswerEvaluatorResult(**x) for x in self._get_existing_output()]
        queries = self._add_retrieved_documents_to_queries(
            queries, self.config.documents_path
        )
        self._add_evaluations_to_answers(queries, evaluations)

        tuples_to_eval = self.__get_tuples_to_evaluate(queries, evaluations)
        if len(tuples_to_eval) == 0:
            return queries

        chunks = [
            tuples_to_eval[i : i + self.config.n_processes]
            for i in range(0, len(tuples_to_eval), self.config.n_processes)
        ]
        pbar = tqdm(
            total=len(tuples_to_eval),
            ncols=100,
            desc="Evaluating answers",
            disable=not use_progress_bar,
            leave=False,
            position=0,
        )

        for chunk in chunks:
            responses = await self._fetch_chunk_pointwise(chunk)
            evaluations.extend(responses)
            pbar.update(len(chunk))
        pbar.close()
        self._add_evaluations_to_answers(queries, evaluations)

        if self.config.verbose:
            print("✅ Done!")
            print(f"Total evaluations: {len(evaluations)}")

        return queries

    async def batch_evaluate_async(self, queries: list[Query]) -> list[Query]:
        if self.config.pairwise:
            return await self._batch_evaluate_pairwise_async(queries)
        return await self._batch_evaluate_pointwise_async(queries)

    def _prepare_queries_for_eval(
        self, queries: list[Query], evaluations: list[dict[str, Any]]
    ) -> list[Query]:
        queries = self._add_retrieved_documents_to_queries(
            queries, self.config.documents_path
        )
        queries = self.__prepare_tuples_for_queries(queries)
        self._add_evaluations_to_answers(queries, evaluations)
        return queries

    def _get_tuples_to_evaluate(
        self, queries: list[Query], evaluations: list[dict[str, Any]]
    ) -> list[tuple[Query, AgentAnswer]]:
        skip_tuples = {(x.qid, x.agent_a, x.agent_b) for x in evaluations}
        tuples_to_eval: list[tuple[Query, AgentAnswer, AgentAnswer]] = []
        all_tuples = 0
        for query in queries:
            for game in query.pairwise_games:
                qid = query.qid
                game_tuple = (qid, game.agent_a, game.agent_b)
                game_tuple_r = (qid, game.agent_b, game.agent_a)
                all_tuples += 1
                if game_tuple in skip_tuples or (
                    game_tuple_r in skip_tuples and self.config.bidirectional
                ):
                    logger.debug(f"Skipping {game_tuple}")
                    continue
                tuples_to_eval.append((query, game.agent_a_answer, game.agent_b_answer))
        if len(tuples_to_eval) == 0:
            logger.info("All answers have been evaluated")
            if self.config.verbose:
                print(
                    f"All {all_tuples} answers are already evaluated.\n"
                    "If you want to re-evaluate them, use the force flag"
                )

        return tuples_to_eval

    async def _batch_evaluate_pairwise_async(self, queries: list[Query]) -> list[Query]:
        use_progress_bar = self.config.use_progress_bar
        evaluations = [AnswerEvaluatorResult(**x) for x in self._get_existing_output()]
        queries = self._prepare_queries_for_eval(queries, evaluations)

        tuples_to_eval = self._get_tuples_to_evaluate(queries, evaluations)
        if len(tuples_to_eval) == 0:
            return queries

        chunks = [
            tuples_to_eval[i : i + self.config.n_processes]
            for i in range(0, len(tuples_to_eval), self.config.n_processes)
        ]
        pbar = tqdm(
            total=len(tuples_to_eval),
            ncols=100,
            desc="Evaluating answers",
            disable=not use_progress_bar,
            leave=False,
            position=0,
        )

        for chunk in chunks:
            responses = await self._fetch_chunk_pairwise(chunk)
            evaluations.extend(responses)
            pbar.update(len(chunk))
        pbar.close()
        self._add_evaluations_to_answers(queries, evaluations)

        if self.config.verbose:
            print("✅ Done!")
            print(f"Total evaluations: {len(evaluations)}")

        return queries

    def __prepare_tuples_for_queries(
        self,
        queries: list[Query],
    ) -> list[Query]:
        for query in queries:
            answers: dict[str, AgentAnswer] = {}
            for agent_answer in query.answers:
                answers[agent_answer.agent] = agent_answer
            random_pairs = self.__generate_agent_pairs(query)
            for agent_a, agent_b in random_pairs:
                query.pairwise_games.append(
                    PairwiseGame(
                        agent_a=agent_a,
                        agent_b=agent_b,
                        agent_a_answer=answers[agent_a],
                        agent_b_answer=answers[agent_b],
                    )
                )
        return queries

    def batch_evaluate(self, queries: list[Query]) -> list[Query]:
        if self.config.pairwise:
            return self._batch_evaluate_pairwise(queries)
        return self._batch_evaluate_pointwise(queries)

    def _batch_evaluate_pointwise(self, queries: list[Query]) -> list[Query]:
        use_progress_bar = self.config.use_progress_bar
        failed_evaluations = 0
        evaluations = [AnswerEvaluatorResult(**x) for x in self._get_existing_output()]
        queries = self._prepare_queries_for_eval(queries, evaluations)
        # queries = self._add_retrieved_documents_to_queries(
        #     queries, self.config.documents_path
        # )
        # self._add_evaluations_to_answers(queries, evaluations)

        tuples_to_eval = self.__get_tuples_to_evaluate(queries, evaluations)
        if len(tuples_to_eval) == 0:
            return queries

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
            evaluation = AnswerEvaluatorResult(
                qid=qid, agent=agent, raw_answer=raw_answer, answer=answer
            )
            q_idx = self.__query_idx[qid]
            a_idx = self.__answer_idx[qid][agent]
            queries[q_idx].answers[a_idx].evaluation = evaluation
            self._dump_response(evaluation, self.output_columns, self.output_file)
        if self.config.verbose:
            print("✅ Done!")
            print(f"Unparsed answers: {failed_evaluations}")
            print(f"Total evaluations: {len(evaluations)}")
        return queries

    def _batch_evaluate_pairwise(self, queries: list[Query]) -> list[Query]:
        use_progress_bar = self.config.use_progress_bar
        failed_evaluations = 0
        all_tuples = 0
        evaluations = [AnswerEvaluatorResult(**x) for x in self._get_existing_output()]
        skip_tuples = {(x.qid, x.agent_a, x.agent_b) for x in evaluations}
        tuples_to_eval: list[tuple[Query, AgentAnswer, AgentAnswer]] = []
        queries = self._prepare_queries_for_eval(queries, evaluations)

        for query in queries:
            for game in query.pairwise_games:
                qid = query.qid
                game_tuple = (qid, game.agent_a, game.agent_b)
                game_tuple_r = (qid, game.agent_b, game.agent_a)
                all_tuples += 1
                if game_tuple in skip_tuples or (
                    game_tuple_r in skip_tuples and self.config.bidirectional
                ):
                    logger.debug(f"Skipping {game_tuple}")
                    continue
                tuples_to_eval.append((query, game.agent_a_answer, game.agent_b_answer))
        if len(tuples_to_eval) == 0:
            logger.info("All answers have been evaluated")
            if self.config.verbose:
                print(
                    f"All {all_tuples} answers are already evaluated.\n"
                    "If you want to re-evaluate them, use the force flag"
                )
            return evaluations
        for query, answer_a, answer_b in tqdm(
            tuples_to_eval,
            desc=use_progress_bar,
            disable=not self.config.verbose,
            leave=False,
            position=0,
            ncols=100,
        ):
            agent_a = answer_a.agent
            agent_b = answer_b.agent
            try:
                raw_answer, parsed_answer = self.evaluate_pairwise(
                    query=query,
                    answer_a=answer_a,
                    answer_b=answer_b,
                    retrieved_documents=query.retrieved_docs,
                )
            except (RetryError, ValueError):
                failed_evaluations += 1
                continue
            evaluation = AnswerEvaluatorResult(
                qid=query.qid,
                agent_a=agent_a,
                agent_b=agent_b,
                raw_answer=raw_answer,
                answer=parsed_answer,
            )
            query_idx = self.__query_idx[query.qid]
            game_idx = self.__pairwise_games_idx[query.qid][(agent_a, agent_b)]
            queries[query_idx].pairwise_games[game_idx].evaluation = evaluation
            self._dump_response(evaluation, self.output_columns, self.output_file)

        if self.config.verbose:
            print("✅ Done!")
            print(f"Unparsed answers: {failed_evaluations}")
            print(f"Total evaluations: {len(evaluations)}")
        return evaluations

    def __generate_agent_pairs(self, query: Query) -> list[tuple[str, str]]:
        """Generates up to self.k random pairs of agents for the given query"""
        query_agents = list({x.agent for x in query.answers})
        # Create all possible pairs
        pairs = [(a, b) for a in query_agents for b in query_agents if a != b]
        if self.config.bidirectional:
            pairs += [(b, a) for a, b in pairs]
        random.shuffle(pairs)
        return pairs[: self.config.k]

    def evaluate(
        self,
        query: Query | str,
        answer: Optional[AgentAnswer | str] = None,
        answer_a: Optional[AgentAnswer | str] = None,
        answer_b: Optional[AgentAnswer | str] = None,
        retrieved_documents: Optional[list[str] | list[Document]] = None,
        document_metadata: Optional[list[dict[str, Any]]] = None,
        query_metadata: Optional[dict[str, Any]] = None,
        answer_metadata: Optional[dict[str, Any]] = None,
        answer_a_metadata: Optional[dict[str, Any]] = None,
        answer_b_metadata: Optional[dict[str, Any]] = None,
    ):
        if self.config.pairwise:
            self._evaluate_pairwise(
                query,
                answer_a,
                answer_b,
                retrieved_documents,
                document_metadata,
                query_metadata,
                answer_a_metadata,
                answer_b_metadata,
            )
        else:
            self._evaluate_pointwise(
                query,
                answer,
                retrieved_documents,
                document_metadata,
                query_metadata,
                answer_metadata,
            )

    def _evaluate_pointwise(
        self,
        query: Query | str,
        answer: AgentAnswer | str,
        retrieved_documents: Optional[list[str] | list[Document]] = None,
        document_metadata: Optional[list[dict[str, Any]]] = None,
        query_metadata: Optional[dict[str, Any]] = None,
        answer_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Any]:
        query = self._assemble_query(query, query_metadata)
        answer = self._assemble_answer(answer, answer_metadata)
        if isinstance(retrieved_documents, str):
            retrieved_documents = [retrieved_documents]
        if retrieved_documents:
            retrieved_and_assembled_docs = self._assemble_documents(
                retrieved_documents, document_metadata
            )
            query.retrieved_docs = retrieved_and_assembled_docs

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

    def _evaluate_pairwise(
        self,
        query: Query | str,
        answer_a: AgentAnswer | str,
        answer_b: AgentAnswer | str,
        retrieved_documents: list[str] | list[Document],
        document_metadata: Optional[list[dict[str, Any]]] = None,
        query_metadata: Optional[dict[str, Any]] = None,
        answer_a_metadata: Optional[dict[str, Any]] = None,
        answer_b_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str]:
        query = self._assemble_query(query, query_metadata)
        answer_a = self._assemble_answer(answer_a, answer_a_metadata)
        answer_b = self._assemble_answer(answer_b, answer_b_metadata)
        if isinstance(retrieved_documents, str):
            retrieved_documents = [retrieved_documents]
        if retrieved_documents:
            retrieved_and_assembled_docs = self._assemble_documents(
                retrieved_documents, document_metadata
            )
            query.retrieved_docs = retrieved_and_assembled_docs

        prompt = self._build_message_pairwise(query, (answer_a, answer_b))
        qid = query.qid
        agent_a_id = answer_a.agent
        agent_b_id = answer_b.agent

        try:
            raw_answer = self.llm_provider(prompt)
        except RetryError as e:
            logger.warning(
                f"Failed to FETCH answers for {qid} {agent_a_id}, {agent_b_id}"
            )
            raise e
        try:
            processed_answer = self._process_answer(raw_answer)
        except ValueError as e:
            logger.warning(
                f"Failed extracting answer for {qid}, {agent_a_id}, {agent_b_id}."
                "Probably not enough tokens in the answer."
                f"Full answer:\n{raw_answer}",
            )
            raise e
        return raw_answer, processed_answer

    def _build_message(
        self, query: Query, answer: AgentAnswer
    ) -> str | list[dict[str, str]]:
        """Builds the message to send to the LLM evaluator"""
        raise NotImplementedError

    def _build_message_pairwise(
        self, query: Query, answer: AgentAnswer | tuple[AgentAnswer, AgentAnswer]
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

    def _prepare_documents(self, query: Query) -> str:
        if len(query.retrieved_docs) == 0:
            return "NO DOCUMENTS WERE RETRIEVED"
        formatted_documents = []
        fields_to_format = [
            field
            for _, field, _, _ in Formatter().parse(self.document_template)
            if field
        ]
        for document in query.retrieved_docs:
            formatter = {}
            if "did" in fields_to_format:
                formatter["did"] = document.did
            if "doc" in fields_to_format:
                formatter["doc"] = document.text
            if "raw_annotation" in fields_to_format:
                formatter["raw_annotation"] = document.evaluation.raw_answer
            if "annotation" in fields_to_format:
                formatter["annotation"] = document.evaluation.answer
            document_metadata = self._get_usable_fields_from_metadata(
                self.document_template, document.metadata
            )
            formatter.update(**document_metadata)
            formatted_documents.append(self.document_template.format(**formatter))
        return "\n".join(formatted_documents)

    def _add_retrieved_documents_to_queries(
        self,
        queries: list[Query],
        documents_path: Optional[str],
        text_column: str = "document_text",
        overwrite: bool = False,
    ):
        if any([len(q.retrieved_docs) > 0 for q in queries]) and not overwrite:
            logger.info(
                "Some queries already have retrieved documents"
                "Refusing to overwrite them."
            )
            return queries
        if documents_path is None:
            logger.warning(
                "No path with retrieved documents provided."
                "Evaluator performance may be affected."
            )
            return queries
        queries_with_docs = load_retrieved_docs_from_csv(
            documents_path, queries, document_text_col=text_column
        )
        return queries_with_docs

    def __build_query_and_answer_idx(self, queries: list[Query]):
        self.__query_idx = {query.qid: idx for idx, query in enumerate(queries)}
        self.__answer_idx: dict[str, dict[str, int]] = {}
        self.__pairwise_games_idx: dict[str, dict[tuple[str, str], int]] = {}
        for query in queries:
            self.__answer_idx[query.qid] = {
                ans.agent: idx for idx, ans in enumerate(query.answers)
            }
            self.__pairwise_games_idx[query.qid] = {
                (game.agent_a, game.agent_b): idx
                for idx, game in enumerate(query.pairwise_games)
            }

    def _add_evaluations_to_answers(
        self, queries: list[Query], evaluations: list[AnswerEvaluatorResult]
    ):
        self.__build_query_and_answer_idx(queries)
        for evaluation in evaluations:
            query_idx = self.__query_idx[evaluation.qid]
            if evaluation.pairwise:
                if not evaluation.agent_a or not evaluation.agent_b:
                    # Should never happen, as the pydantic model enforces this
                    raise ValueError("Pairwise evaluations require two agents")
                agents = (evaluation.agent_a, evaluation.agent_b)
                if agents not in self.__pairwise_games_idx[evaluation.qid]:
                    agents = (evaluation.agent_b, evaluation.agent_a)
                    if agents not in self.__pairwise_games_idx[evaluation.qid]:
                        raise ValueError(
                            f"Pairwise evaluation between {evaluation.agent_a} and {evaluation.agent_b} "
                            f"not found in query {evaluation.qid}"
                        )
                game_idx = self.__pairwise_games_idx[evaluation.qid][agents]
                queries[query_idx].pairwise_games[game_idx].evaluation = evaluation

            else:
                if evaluation.agent is None:
                    # Should never happen.
                    raise ValueError("Evaluation must have an agent")
                answer_idx = self.__answer_idx[evaluation.qid][evaluation.agent]
                queries[query_idx].answers[answer_idx].evaluation = evaluation


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
            valid_keys = [field for field in type_config.get_model_fields()]
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
