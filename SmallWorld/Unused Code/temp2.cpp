// Returns the biconnected components of the masked graph.
inline std::vector<std::vector<v_size_t>> bicomponents(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask)
{
	v_size_t vct = adj.size();
	v_size_t numI = 0;
	std::vector<v_size_t> num(vct, 0);
	std::vector<v_size_t> low(vct, 0);
	std::vector<std::vector<v_size_t>> bicomps;
	std::vector<std::pair<v_size_t, v_size_t>> edgeStack;
	std::vector<mask_t> inBicomp(vct, false);

	const auto visitStart = [&numI, &num, &low, &edgeStack](v_size_t u, v_size_t v) -> void
	{
		low[v] = num[v] = ++numI;
		if (u != v)
			edgeStack.emplace_back(u, v);
	};

	const auto visitEnd = [&num, &low, &bicomps, &edgeStack, &inBicomp](v_size_t u, v_size_t v) -> void
	{
		if (u == v)
			return;
		low[u] = std::min(low[u], low[v]);
		// If v is an articulation point, add a new biconnected component.
		if (low[v] < num[u])
			return;

		std::vector<v_size_t> curBicomp;
		std::pair<v_size_t, v_size_t> edge;

		// For each edge on the stack until u-v, add it to the current bicomponent if not already added
		// edge.first will be added as edge.second lower on the stack.
		while (edge = edgeStack.back(), edgeStack.pop_back(), num[edge.first] >= num[v])
			if (!inBicomp[edge.second])
			{
				curBicomp.push_back(edge.second);
				inBicomp[edge.second] = true;
			}

		assert(edge == std::make_pair(u, v));

		if (!inBicomp[edge.second])
			curBicomp.push_back(edge.second);

		if (!inBicomp[edge.first])
			curBicomp.push_back(edge.first);

		// Reset inBicomp for the next bicomponent.
		for (v_size_t x : curBicomp)
			inBicomp[x] = false;

		// Add the current biconnected component to the list.
		bicomps.emplace_back(std::move(curBicomp));
	};

	const auto isVisited = [&num](v_size_t w) -> bool
	{
		return num[w] != 0;
	};

	const auto visitAgain = [&num, &low, &edgeStack](v_size_t u, v_size_t v, v_size_t w) -> void
	{
		if (w == u || num[w] >= num[v]) return;
		edgeStack.emplace_back(v, w);
		low[v] = std::min(low[v], num[w]);
	};

	DFS dfs(visitStart, visitEnd, isVisited, visitAgain);
	dfs(adj, mask);

	return bicomps;
}