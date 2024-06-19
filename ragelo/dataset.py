import json
import pickle
from typing import Dict, Optional, Union

from ragelo.evaluators.answer_evaluators import BaseAnswerEvaluator
from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator
from ragelo.logger import logger
from ragelo.types.types import AgentAnswer, BaseModel, Document, Query


class QueriesDataset(BaseModel):
    """A Dataset is a collection of Queries."""

    queries: Dict[str, Query] = {}

    # Override the __getitem__ method to return the query by qid
    def __getitem__(self, qid: str) -> Query:
        return self.queries[qid]

    # Override the __setitem__ method to add a query by qid
    def __setitem__(self, qid: str, query: Union[Query, str]):
        if isinstance(query, str):
            query = Query(qid=qid, query=query)
        else:
            if qid != query.qid:
                logger.warning(
                    f"qid {qid} is provided, but the query has a different qid {query.qid}.",
                    f"I will not override the query, but you can access it with query[{qid}].",
                )
        if qid in self.queries:
            logger.warning(
                f"Query with qid {qid} already exists in dataset. Overwriting."
            )
        self.queries[qid] = query

    # override __contains__ to check if a query exists in the dataset
    def __contains__(self, qid: str) -> bool:
        return qid in self.queries

    # override __len__ to return the number of queries in the dataset
    def __len__(self) -> int:
        return len(self.queries)

    # override __iter__ to return an iterator over the queries in the dataset
    def __iter__(self):
        return iter(self.queries.values())

    def keys(self):
        return self.queries.keys()

    def values(self):
        return self.queries.values()

    def items(self):
        return self.queries.items()

    def qids(self):
        return self.keys()

    def documents_iter(self):
        for q in self.queries.values():
            for doc in q.retrieved_docs:
                yield doc

    def add_retrieved_doc(
        self, qid: str, doc: Union[Document, str], doc_id: Optional[str] = None
    ):
        if qid not in self.queries:
            raise KeyError(f"Query with qid {qid} does not exist in dataset.")
        self.queries[qid].add_retrieved_doc(doc, doc_id)

    def add_agent_answer(
        self, qid: str, answer: Union[AgentAnswer, str], agent: Optional[str] = None
    ):
        if qid not in self.queries:
            raise KeyError(f"Query with qid {qid} does not exist in dataset.")
        self.queries[qid].add_agent_answer(answer, agent)

    def batch_evaluate(
        self, evaluator: Union[BaseRetrievalEvaluator, BaseAnswerEvaluator]
    ):
        queries = list(self.queries.values())
        evaluated_queries = evaluator.batch_evaluate(queries)
        for q in evaluated_queries:
            self.queries[q.qid] = q

    def save(self, path: str):
        if path.endswith(".pkl"):
            with open(path, "wb") as f:
                pickle.dump(self, f)
        elif path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(self.dict(), f)
        else:
            raise ValueError(f"Format {path} not supported")

    @classmethod
    def load(cls, path: str):
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        elif path.endswith(".json"):
            with open(path, "r") as f:
                return cls(**json.load(f))
