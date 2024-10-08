"""Base models for ranking a set of answers generated by LLMs"""

import csv
from typing import Dict, List, Optional, Tuple, Type, get_type_hints

from ragelo.logger import logger
from ragelo.types.configurations.agent_ranker_configs import AgentRankerConfig
from ragelo.types.types import Query
from ragelo.utils import load_answer_evaluations_from_csv


class AgentRanker:
    config: AgentRankerConfig

    def __init__(
        self,
        config: AgentRankerConfig,
    ):
        self.config = config
        self.name = self.config.ranker_name
        self.agents_evaluations_file = self.config.agents_evaluations_file

    def run(
        self,
        queries: Optional[List[Query]] = None,
        evaluations_file: Optional[str] = None,
    ) -> Dict[str, int]:
        """Compute score for each agent"""
        raise NotImplementedError

    def _prepare_queries(
        self,
        queries: Optional[List[Query]] = None,
        evaluations_file: Optional[str] = None,
    ) -> List[Query]:
        if queries is None:
            if evaluations_file is None:
                raise ValueError(
                    "Either queries or evaluations_file should be provided"
                )
            queries = load_answer_evaluations_from_csv(evaluations_file)
        return queries

    @classmethod
    def from_config(cls, config: AgentRankerConfig):
        return cls(config)
        # evaluations = cls.load_evaluations(config.evaluations_file)
        # return cls(config, evaluations)

    def get_agents_ratings(self) -> Dict[str, float]:
        """Returns the score of all players"""
        raise NotImplementedError

    def dump_ranking(self):
        if not self.config.write_output:
            return
        with open(self.agents_evaluations_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["agent", "score"])
            for agent, rating in sorted(
                self.get_agents_ratings().items(), key=lambda x: x[1], reverse=True
            ):
                writer.writerow([agent, rating])

    def print_ranking(self):
        if not self.config.verbose:
            return
        scores = sorted(
            self.get_agents_ratings().items(), key=lambda x: x[1], reverse=True
        )
        if self.config.rich_print:
            try:
                import rich

                rich.print(
                    f"-------[bold white] Agent Scores by {self.name} [/bold white]-------"
                )

                for agent, rating in scores:
                    rich.print(f"[bold white]{agent:<15}[/bold white]: {rating:.1f}")
            except ImportError:
                logger.warning("Rich not installed. Using plain print")
                self.config.rich_print = False
        if not self.config.rich_print:
            print(f"------- Agent Scores by {self.name} -------")
            for agent, rating in scores:
                print(f"{agent:<15}: {rating:.1f}")

    @classmethod
    def get_config_class(cls) -> Type[AgentRankerConfig]:
        return get_type_hints(cls)["config"]

    def _flatten_evaluations(self, queries) -> List[Tuple[str, str, str]]:
        evaluations = []
        for q in queries:
            for game in q.pairwise_games:
                evaluations.append(
                    (
                        game.agent_a_answer.agent,
                        game.agent_b_answer.agent,
                        game.evaluation.answer,
                    )
                )
        return evaluations


class AgentRankerFactory:
    registry: Dict[str, Type[AgentRanker]] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type[AgentRanker]):
            if name in cls.registry:
                logger.warning(f"Overwriting {name} in Answer Evaluator registry")
            cls.registry[name.lower()] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        ranker_name: str,
        config: Optional[AgentRankerConfig] = None,
        **kwargs,
    ) -> AgentRanker:
        if ranker_name.lower() not in cls.registry:
            raise ValueError(f"Unknown Agent Ranker {ranker_name}")
        if config is None:
            class_ = cls.registry[ranker_name.lower()]
            type_config = class_.get_config_class()
            valid_keys = [field for field in type_config.get_model_fields()]
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            config = type_config(**valid_args)
        return cls.registry[ranker_name.lower()].from_config(config)


def get_agent_ranker(
    ranker_name: str,
    config: Optional[AgentRankerConfig] = None,
    **kwargs,
) -> AgentRanker:
    return AgentRankerFactory.create(ranker_name, config, **kwargs)
