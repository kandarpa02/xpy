from .base import Tensor

def assign_names(topo):
	names = {}
	temp_i = 0
	for n in topo:
		if n.parents == ():
				# Leaf nodes get parameter names
			names[n] = f"x{n.index}"
		else:
			names[n] = f"t{temp_i}"
			temp_i += 1
	return names

# def collect_leaves(roots):
# 	seen = set()
# 	leaves = []

# 	def visit(n):
# 		if n in seen:
# 				return
# 		seen.add(n)

# 		if n.parents == ():
# 				leaves.append(n)
# 		else:
# 				for p in n.parents:
# 						visit(p)

# 	for r in roots:
# 			visit(r)

# 	return leaves

def collect_leaves(roots):
    seen = set()
    leaves = []

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))

        if n.parents == ():
            leaves.append(n)
        else:
            for p in n.parents:
                visit(p)

    for r in roots:
        visit(r)

    return leaves

# def auto_index_leaves(roots):
# 	leaves = collect_leaves(roots)
# 	for i, leaf in enumerate(leaves):
# 		leaf.index = i

def auto_index_leaves(roots):
    leaves = collect_leaves(roots)
    for i, leaf in enumerate(leaves):
        leaf.index = i

def topo_sort(roots):
	auto_index_leaves(roots)
	seen = set()
	order = []

	def visit(n):
		if n in seen:
			return
		seen.add(n)
		for p in n.parents:
			visit(p)
		order.append(n)

	for r in roots:
		visit(r)

	return order
