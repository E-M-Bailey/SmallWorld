// Returns the connected components
inline std::vector<std::vector<v_size_t>> components(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask)
{
	v_size_t vct = static_cast<v_size_t>(adj.size());
	std::vector<mask_t> visited(vct, false);
	std::vector<std::vector<v_size_t>> comps;

	const auto visitStart = [&visited, &comps](v_size_t u, v_size_t v) -> void
	{
		visited[v] = true;
		if (u == v)
			comps.emplace_back();
		comps.back().push_back(v);
	};

	const auto visitEnd = [](v_size_t u, v_size_t v) -> void {};

	const auto isVisited = [&visited](v_size_t v) -> bool
	{
		return visited[v];
	};

	const auto visitAgain = [](v_size_t u, v_size_t v, v_size_t w) -> void {};

	DFS dfs(visitStart, visitEnd, isVisited, visitAgain);
	dfs(adj, mask);

	return comps;
}