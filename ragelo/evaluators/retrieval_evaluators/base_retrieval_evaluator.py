"""A Retrieval Evaluator is a class that evaluates the results of a retrieval system.
It receives a set of queries used to retrieve a document and their respective retrieved documents,
and returns a score or a label for each document."""

import asyncio
from typing import Any, Callable, Optional, Type, get_type_hints

from aiohttp import ClientSession
from tenacity import RetryError
from tqdm.auto import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types import (
    AnswerFormat,
    Document,
    Query,
    RetrievalEvaluatorResult,
    RetrievalEvaluatorTypes,
)
from ragelo.types.configurations import BaseRetrievalEvaluatorConfig


class BaseRetrievalEvaluator(BaseEvaluator):
    config: BaseRetrievalEvaluatorConfig
    output_columns: list[str] = ["qid", "did", "raw_answer", "answer"]
    output_file: str = "retrieval_evaluations.csv"

    def __init__(
        self,
        config: BaseRetrievalEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider
        self.query_idx: dict[str, int] = {}
        self.doc_idx: dict[str, dict[str, int]] = {}

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
        self, queries: list[Query], answers: list[RetrievalEvaluatorResult]
    ) -> list[tuple[Query, Document]]:
        skip_docs = {(x.qid, x.did) for x in answers}
        tuples_to_eval = []
        all_tuples = 0
        for query in queries:
            for document in query.retrieved_docs:
                qid = query.qid
                did = document.did
                all_tuples += 1
                if (qid, did) in skip_docs:
                    logger.debug(f"Skipping {qid} {did}")
                    continue
                tuples_to_eval.append((query, document))
        if len(tuples_to_eval) == 0:
            logger.info("All documents have been evaluated")
            if self.config.verbose:
                print(
                    f"All {all_tuples} documents are already evaluated.\n"
                    "If you want to re-evaluate documents, use the --force flag."
                )

        return tuples_to_eval

    async def __fetch_chunk(
        self, chunk: list[tuple[Query, Document]]
    ) -> list[RetrievalEvaluatorResult]:
        qids = []
        dids = []
        async with ClientSession() as session:
            tasks = []
            for query, document in chunk:
                message = self._build_message(query, document)
                qids.append(query.qid)
                dids.append(document.did)
                tasks.append(self.llm_provider.call_async(message, session))
            raw_answers = await asyncio.gather(*tasks)
            parsed_answers = []
            for qid, did, raw_answer in zip(qids, dids, raw_answers):
                try:
                    answer = self._process_answer(raw_answer)
                except ValueError:
                    logger.warning(f"Failed to PARSE answer for qid: {qid} did: {did}")
                    continue
                parsed_answers.append(
                    RetrievalEvaluatorResult(
                        qid=qid,
                        did=did,
                        raw_answer=raw_answer,
                        answer=answer,
                    )
                )
                self._dump_response(
                    parsed_answers[-1], self.output_columns, self.output_file
                )
        return parsed_answers

    async def batch_evaluate_async(self, queries: list[Query]) -> list[Query]:
        """Evaluate all the documents for a list of queries"""
        use_progress_bar = self.config.use_progress_bar
        answers = [RetrievalEvaluatorResult(**x) for x in self._get_existing_output()]
        self._add_evaluations_to_documents(queries, answers)
        tuples_to_eval = self.__get_tuples_to_evaluate(queries, answers)
        if len(tuples_to_eval) == 0:
            return queries

        chunks = [
            tuples_to_eval[i : i + self.config.n_processes]
            for i in range(0, len(tuples_to_eval), self.config.n_processes)
        ]
        pbar = tqdm(
            total=len(tuples_to_eval),
            ncols=100,
            desc="Evaluating documents",
            disable=not use_progress_bar,
            leave=False,
            position=0,
        )

        for chunk in chunks:
            responses = await self.__fetch_chunk(chunk)
            answers.extend(responses)
            pbar.update(len(chunk))
        pbar.close()
        self._add_evaluations_to_documents(queries, answers)

        if self.config.verbose:
            print("✅ Done!")
            print(f"Total evaluations: {len(answers)}")

        return queries

    def __build_query_and_doc_idx(self, queries: list[Query]):
        self.__query_idx = {query.qid: idx for idx, query in enumerate(queries)}
        self.__doc_idx: dict[str, dict[str, int]] = {}
        for query in queries:
            self.__doc_idx[query.qid] = {
                doc.did: idx for idx, doc in enumerate(query.retrieved_docs)
            }

    def _add_evaluations_to_documents(
        self, queries: list[Query], evaluations: list[RetrievalEvaluatorResult]
    ):
        self.__build_query_and_doc_idx(queries)

        for evaluation in evaluations:
            if evaluation.qid not in self.__query_idx:
                raise ValueError(f"Query {evaluation.qid} not found in the queries")
            if evaluation.did not in self.__doc_idx[evaluation.qid]:
                raise ValueError(
                    f"Document {evaluation.did} not found in the documents retrieved for query {evaluation.qid}"
                )
            q_idx = self.__query_idx[evaluation.qid]
            d_idx = self.__doc_idx[evaluation.qid][evaluation.did]
            queries[q_idx].retrieved_docs[d_idx].evaluation = evaluation

    def batch_evaluate(self, queries: list[Query]) -> list[Query]:
        """Evaluate all the documents for a list of queries"""
        use_progress_bar = self.config.verbose
        answers = [RetrievalEvaluatorResult(**x) for x in self._get_existing_output()]
        self._add_evaluations_to_documents(queries, answers)

        failed_evaluations = 0
        tuples_to_eval = self.__get_tuples_to_evaluate(queries, answers)
        all_tuples = 0
        if len(tuples_to_eval) == 0:
            logger.info("All documents have been evaluated")
            if self.config.verbose:
                print(
                    f"All {all_tuples} documents are already evaluated.\n"
                    "If you want to re-evaluate documents, use the --force flag."
                )
            return queries
        for query, document in tqdm(
            tuples_to_eval,
            desc="Evaluating retrieved documents",
            disable=not use_progress_bar,
            ncols=100,
            # leave=False,
            position=0,
        ):
            qid = query.qid
            did = document.did
            try:
                raw_answer, answer = self.evaluate(query, document)
            except (RetryError, ValueError):
                failed_evaluations += 1
                continue

            answer = RetrievalEvaluatorResult(
                qid=qid,
                did=did,
                raw_answer=raw_answer,
                answer=answer,
            )

            # add the evaluations to the Document objects
            q_idx = self.__query_idx[qid]
            d_idx = self.__doc_idx[qid][did]
            queries[q_idx].retrieved_docs[d_idx].evaluation = answer
            self._dump_response(answer, self.output_columns, self.output_file)

        if self.config.verbose:
            print("✅ Done!")
            print(f"Unparsed answers: {failed_evaluations}")
            print(f"Total evaluations: {len(answers)}")
        return queries

    def evaluate(
        self,
        query: Query | str,
        document: Document | str,
        query_metadata: Optional[dict[str, Any]] = None,
        doc_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Any]:
        """Evaluates a single query-document pair. Returns the raw answer and the processed answer."""
        if isinstance(query, str):
            query = Query(qid="<no_qid>", query=query)
        if isinstance(document, str):
            document = Document(did="<no_did>", text=document)
        query.add_metadata(query_metadata)
        document.add_metadata(doc_metadata)

        message = self._build_message(query, document)
        try:
            raw_answer = self.llm_provider(message)
        except RetryError as e:
            logger.warning(
                f"Failed to FETCH answers for qid: {query.qid} did: {document.did}"
            )
            raise e
        try:
            answer = self._process_answer(raw_answer)
        except ValueError as e:
            logger.warning(
                f"Failed to PARSE answer for qid: {query.qid} did: {document.did}"
            )
            raise e
        return raw_answer, answer

    def _build_message(
        self, query: Query, document: Document
    ) -> str | list[dict[str, str]]:
        """Builds the prompt to send to the LLM."""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls, config: BaseRetrievalEvaluatorConfig, llm_provider: BaseLLMProvider
    ):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> Type[BaseRetrievalEvaluatorConfig]:
        return get_type_hints(cls)["config"]

    @staticmethod
    def _construct_list_of_answers(
        answers: list[dict[str, str]]
    ) -> list[RetrievalEvaluatorResult]:
        return [RetrievalEvaluatorResult(**x) for x in answers]


class RetrievalEvaluatorFactory:
    registry: dict[RetrievalEvaluatorTypes | str, Type[BaseRetrievalEvaluator]] = {}

    @classmethod
    def register(cls, evaluator_name: RetrievalEvaluatorTypes) -> Callable:
        def inner_wrapper(
            wrapped_class: Type[BaseRetrievalEvaluator],
        ) -> Type[BaseRetrievalEvaluator]:
            if evaluator_name in cls.registry:
                logger.debug(f"Overwriting {evaluator_name} in registry")
            cls.registry[evaluator_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        evaluator_name: RetrievalEvaluatorTypes | str,
        llm_provider: BaseLLMProvider | str,
        config: Optional[BaseRetrievalEvaluatorConfig] = None,
        **kwargs,
    ) -> BaseRetrievalEvaluator:
        if isinstance(llm_provider, str):
            llm_provider_instance = get_llm_provider(llm_provider, **kwargs)
        else:
            llm_provider_instance = llm_provider
        if evaluator_name not in cls.registry:
            raise ValueError(
                f"Unknown retrieval evaluator {evaluator_name}\n"
                f"Valid options are {list(cls.registry.keys())}"
            )
        if config is None:
            class_ = cls.registry[evaluator_name]
            type_config = class_.get_config_class()
            valid_keys = [field for field in type_config.get_model_fields()]
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            config = type_config(**valid_args)
        return cls.registry[evaluator_name].from_config(config, llm_provider_instance)


def get_retrieval_evaluator(
    evaluator_name: RetrievalEvaluatorTypes | str,
    llm_provider: BaseLLMProvider | str,
    config: Optional[BaseRetrievalEvaluatorConfig] = None,
    **kwargs,
) -> BaseRetrievalEvaluator:
    return RetrievalEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )
