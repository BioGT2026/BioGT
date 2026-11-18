"""Personalized PageRank baseline for disease-biomarker link prediction."""

from collections import defaultdict

from Model.network_propagation_model import NetworkPropagationModel


class PPRModel(NetworkPropagationModel):
    """Personalized PageRank baseline.

    This is an explicit alias around the non-parametric propagation baseline so
    experiments can report PPR separately from other propagation variants.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_mode = "ppr"

    def _propagate_from_source(self, g, source_key):
        graph_key = self._graph_cache_key(g)
        cache_key = (graph_key, "ppr", source_key, self.restart_prob, self.max_steps)
        if cache_key in self._propagation_cache:
            return self._propagation_cache[cache_key]

        adjacency = self._build_adjacency(g)
        restart = min(max(self.restart_prob, 0.0), 1.0)
        continue_prob = 1.0 - restart
        current = {source_key: 1.0}

        for _ in range(max(0, self.max_steps)):
            next_scores = defaultdict(float)
            next_scores[source_key] += restart
            dangling_mass = 0.0

            for node_key, mass in current.items():
                neighbors = adjacency.get(node_key, [])
                if not neighbors:
                    dangling_mass += mass
                    continue
                share = continue_prob * mass / float(len(neighbors))
                for neighbor in neighbors:
                    next_scores[neighbor] += share

            if dangling_mass:
                next_scores[source_key] += continue_prob * dangling_mass
            current = dict(next_scores)

        self._propagation_cache[cache_key] = current
        return current
