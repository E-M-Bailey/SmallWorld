#ifndef DFS_H
#define DFS_H

#include <vector>

// visitStart and visitEnd are passed u and v.
// isVisited is passed the vertex in question.
// visitAgain is passed u, v, and w.
// u == v iff v is the first vertex in its connected component (as adj has no self-loops).
// All methods have void return type except isVisited, which returns a bool corresponding to whether a vertex is already visited.
template<typename VisitStart, typename VisitEnd, typename IsVisited, typename VisitAgain>
class DFS
{
public:
	const VisitStart& visitStart;
	const VisitEnd& visitEnd;
	const IsVisited& isVisited;
	const VisitAgain& visitAgain;

	inline DFS(const VisitStart& visitStart, const VisitEnd& visitEnd, const IsVisited& isVisited, const VisitAgain& visitAgain) :
		visitStart(visitStart),
		visitEnd(visitEnd),
		isVisited(isVisited),
		visitAgain(visitAgain)
	{}

private:
	inline void visit(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask, v_size_t u, v_size_t v)
	{
		visitStart(u, v);
		for (v_size_t w : adj[v])
		{
			if (!mask[w]) continue;

			if (isVisited(w))
				visitAgain(u, v, w);
			else
				// Stack overflow is possible here if the stack size is too small.
				visit(adj, mask, v, w);
		}
		visitEnd(u, v);
	}

public:
	inline void operator()(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask)
	{
		v_size_t vct = adj.size();
		for (v_size_t s = 0; s < vct; s++)
			if (mask[s] && !isVisited(s))
				visit(adj, mask, s, s);
	}
};

#endif
