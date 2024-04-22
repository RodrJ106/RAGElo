"""Base model for dealing with answer evaluators"""

import asyncio
import itertools
import random
from string import Formatter
from typing import Any, Callable, Optional, Type, get_type_hints

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

EvalT = tuple[Query, AgentAnswer | PairwiseGame]
MetadataT = dict[str, Any]


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
                self.config.scoring_keys = [config.scoring_key]
            else:
                self.config.scoring_keys = config.scoring_key
            self.output_columns = [
                "qid",
                "agent",
                "raw_answer",
            ] + self.config.scoring_keys
        if config.scoring_key and config.scoring_key not in self.output_columns:
            print(f"Adding scoring key {config.scoring_key} to output columns")
            self.output_columns.append(self.config.scoring_key)
        if config.scoring_keys:
            missing_keys = [
                key for key in config.scoring_keys if key not in self.output_columns
            ]
            self.output_columns.extend(missing_keys)

    async def __async_evaluate(self, eval_sample: EvalT) -> AnswerEvaluatorResult:
        query, agent_answer = eval_sample
        pairwise = False
        exc = None
        if isinstance(agent_answer, AgentAnswer):
            prompt = self._build_message(query, agent_answer)
        else:
            prompt = self._build_message_pairwise(
                query, (agent_answer.agent_a_answer, agent_answer.agent_b_answer)
            )
            pairwise = True
        try:
            raw_answer = await self.llm_provider.call_async(prompt)
        except Exception as e:
            logger.warning(f"Failed to FETCH answers for qid: {query.qid}")
            if pairwise:
                logger.warning("agents: {agent_answer.agent_a}, {agent_answer.agent_b}")
            else:
                logger.warning("agent: {agent_answer.agent}")
            logger.warning(f"error: {e}")
            exc = str(e)
        try:
            answer = self._process_answer(raw_answer)
        except ValueError as e:
            if pairwise:
                logger.warning(
                    f"Failed to PARSE answer for qid: {query.qid} agents: {agent_answer.agent_a}, {agent_answer.agent_b}\n"
                    f"Raw answer: {raw_answer}"
                )
            else:
                logger.warning(
                    f"Failed to PARSE answer for qid: {query.qid} agent: {agent_answer.agent}\n"
                    f"Raw answer: {raw_answer}"
                )
            answer = None
            exc = str(e)
        if pairwise:
            ans = AnswerEvaluatorResult(
                qid=query.qid,
                agent_a=agent_answer.agent_a,
                agent_b=agent_answer.agent_b,
                raw_answer=raw_answer,
                answer=answer,
                exception=exc,
            )
        else:
            ans = AnswerEvaluatorResult(
                qid=query.qid,
                agent=agent_answer.agent,
                raw_answer=raw_answer,
                answer=answer,
                exception=exc,
            )
        self._dump_response(ans, self.output_columns, self.output_file)
        return ans

    def __prepare_queries(
        self, queries: list[Query], evaluations: list[AnswerEvaluatorResult]
    ) -> list[Query]:
        # check if we need to load the retrieved documents
        add_docs = False
        queries_with_docs = []
        if all(len(x.retrieved_docs) == 0 for x in queries):
            add_docs = True
            queries_with_docs = load_retrieved_docs_from_csv(
                self.config.documents_path, queries
            )
        evaluations_dict: dict[
            str, dict[str | tuple[str, str], AnswerEvaluatorResult]
        ] = {}
        # preprocess evaluations for easier access later
        agent: str | tuple[str, str]
        for evaluation in evaluations:
            if evaluation.pairwise:
                if not evaluation.agent_a or not evaluation.agent_b:
                    # Should never happen, as the pydantic model enforces this
                    raise ValueError("Pairwise evaluations require two agents")
                agent = (evaluation.agent_a, evaluation.agent_b)
            else:
                if evaluation.agent is None:
                    # Should never happen.
                    raise ValueError("Evaluation must have an agent")
                agent = evaluation.agent
            if evaluation.qid not in evaluations_dict:
                evaluations_dict[evaluation.qid] = {}
            evaluations_dict[evaluation.qid][agent] = evaluation

        for q_idx, query in enumerate(queries):
            # check if the query has a list of retrieved documents
            if add_docs and len(queries_with_docs[q_idx].retrieved_docs) > 0:
                if len(queries[q_idx].retrieved_docs) == 0 or self.config.force:
                    queries[q_idx].retrieved_docs = queries_with_docs[
                        q_idx
                    ].retrieved_docs
            # load existing evaluations for agent answers
            answers = {}
            for agent_answer in query.answers:
                answers[agent_answer.agent] = agent_answer
                existing_evaluation = evaluations_dict.get(query.qid, {}).get(
                    agent_answer.agent
                )
                agent_answer.evaluation = existing_evaluation

            # if pairwise, also load existing evaluations for pairwise games
            if self.config.pairwise:
                random_pairs = self.__generate_agent_pairs(query)
                for agent_a, agent_b in random_pairs:
                    # Check if the evaluation already exists
                    existing_evaluation = evaluations_dict.get(query.qid, {}).get(
                        (agent_a, agent_b)
                    )
                    query.pairwise_games.append(
                        PairwiseGame(
                            agent_a=agent_a,
                            agent_b=agent_b,
                            agent_a_answer=answers[agent_a],
                            agent_b_answer=answers[agent_b],
                            evaluation=existing_evaluation,
                        )
                    )

        return queries

    def __get_tuples_to_evaluate(
        self,
        queries: list[Query],
        evaluations: list[AnswerEvaluatorResult],
    ) -> list[EvalT]:
        skip_tuples = {
            (x.qid, x.agent) if x.pairwise else (x.qid, x.agent_a, x.agent_b)
            for x in evaluations
        }

        if self.config.pairwise:
            skip_tuples = {(x.qid, x.agent_a, x.agent_b) for x in evaluations}
        else:
            skip_tuples = {(x.qid, x.agent) for x in evaluations}
        tuples_to_eval: list[EvalT] = []
        all_tuples = 0
        for query in queries:
            if self.config.pairwise:
                for game in query.pairwise_games:
                    qid = query.qid
                    game_tuple = (qid, game.agent_a, game.agent_b)
                    all_tuples += 1
                    if game_tuple in skip_tuples:
                        logger.debug(f"Skipping {game_tuple}")
                        continue
                    tuples_to_eval.append((query, game))
            else:
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

    async def batch_evaluate(self, queries: list[Query]) -> list[Query]:
        use_progress_bar = self.config.use_progress_bar
        evaluations = [AnswerEvaluatorResult(**x) for x in self._get_existing_output()]
        queries = self.__prepare_queries(queries, evaluations)
        tuples_to_eval = self.__get_tuples_to_evaluate(queries, evaluations)
        if len(tuples_to_eval) == 0:
            return queries

        pbar = tqdm(
            total=len(tuples_to_eval),
            ncols=100,
            desc="Evaluating answers",
            disable=not use_progress_bar,
            leave=False,
            position=0,
        )
        aws_ended = False
        pending = set()
        aws = map(self.__async_evaluate, tuples_to_eval)
        aws = iter(aws)
        # while there are pending tasks or not all tasks are done
        while pending or not aws_ended:
            # while there are less than n_processes pending tasks
            while len(pending) < self.config.n_processes and not aws_ended:
                try:
                    aw = next(aws)
                except StopIteration:
                    aws_ended = True  # all tasks have been scheduled
                else:
                    pending.add(asyncio.ensure_future(aw))
            if not pending:
                break

            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            while done:
                evaluation = await done.pop()
                pbar.update()
                if evaluation.exception:
                    continue
                evaluations.append(evaluation)

        pbar.close()
        self.__add_evaluations_to_answers(queries, evaluations)
        if self.config.verbose:
            print("✅ Done!")
            print(f"Total evaluations: {len(evaluations)}")
        return queries

    def __generate_agent_pairs(self, query: Query) -> list[tuple[str, str]]:
        """Generates up to self.k random pairs of agents for the given query"""
        query_agents = list({x.agent for x in query.answers})
        # Create all possible pairs
        pairs = list(itertools.combinations(query_agents, 2))
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
        document_metadata: Optional[list[MetadataT]] = None,
        query_metadata: Optional[MetadataT] = None,
        answer_metadata=None,
        answer_a_metadata: Optional[MetadataT] = None,
        answer_b_metadata: Optional[MetadataT] = None,
    ):
        query = self._assemble_query(query, query_metadata)
        if isinstance(retrieved_documents, str):
            retrieved_documents = [retrieved_documents]
        if retrieved_documents:
            retrieved_and_assembled_docs = self._assemble_documents(
                retrieved_documents, document_metadata
            )
            query.retrieved_docs = retrieved_and_assembled_docs

        if self.config.pairwise:
            if not answer_a or not answer_b:
                raise ValueError("Pairwise evaluations require two answers")
            answer_a = self._assemble_answer(answer_a, answer_a_metadata)
            answer_b = self._assemble_answer(answer_b, answer_b_metadata)
            game = PairwiseGame(
                agent_a_answer=answer_a,
                agent_b_answer=answer_b,
            )
            result = asyncio.run(self.__async_evaluate((query, game)))
        else:
            if not answer:
                raise ValueError("Pointwise evaluations require an answer")
            answer = self._assemble_answer(answer, answer_metadata)
            result = asyncio.run(self.__async_evaluate((query, answer)))
        return result.raw_answer, result.answer

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
            if "raw_annotation" in fields_to_format and document.evaluation:
                formatter["raw_annotation"] = document.evaluation.raw_answer
            if "annotation" in fields_to_format and document.evaluation:
                formatter["annotation"] = str(document.evaluation.answer)
            document_metadata = self._get_usable_fields_from_metadata(
                self.document_template, document.metadata
            )
            formatter.update(**document_metadata)
            formatted_documents.append(self.document_template.format(**formatter))
        return "\n".join(formatted_documents)

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

    def __add_evaluations_to_answers(
        self, queries: list[Query], evaluations: list[AnswerEvaluatorResult]
    ) -> list[Query]:
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
                        logger.warning(
                            f"Pairwise evaluation between {evaluation.agent_a} and {evaluation.agent_b} "
                            f"not found in query {evaluation.qid}"
                        )
                        continue

                game_idx = self.__pairwise_games_idx[evaluation.qid][agents]
                queries[query_idx].pairwise_games[game_idx].evaluation = evaluation

            else:
                if evaluation.agent is None:
                    # Should never happen.
                    raise ValueError("Evaluation must have an agent")
                answer_idx = self.__answer_idx[evaluation.qid][evaluation.agent]
                queries[query_idx].answers[answer_idx].evaluation = evaluation
        return queries

    @classmethod
    def from_config(
        cls, config: BaseAnswerEvaluatorConfig, llm_provider: BaseLLMProvider
    ):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> Type[BaseAnswerEvaluatorConfig]:
        return get_type_hints(cls)["config"]


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
