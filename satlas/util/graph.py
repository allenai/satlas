from satlas.util import geom
from satlas.util import grid_index

import math
import numpy

class Vertex(object):
	def __init__(self, id, point):
		self.id = id
		self.point = point
		self.in_edges = []
		self.out_edges = []

	def _neighbors(self):
		n = {}
		for edge in self.in_edges:
			n[edge.src] = edge
		for edge in self.out_edges:
			n[edge.dst] = edge
		return n

	def neighbors(self):
		return self._neighbors().keys()

	def __repr__(self):
		return 'Vertex({}, {}, {} in {} out)'.format(self.id, self.point, len(self.in_edges), len(self.out_edges))

class Edge(object):
	def __init__(self, id, src, dst):
		self.id = id
		self.src = src
		self.dst = dst

	def bounds(self):
		return self.src.point.bounds().extend(self.dst.point)

	def segment(self):
		return geom.Segment(self.src.point, self.dst.point)

	def closest_pos(self, point):
		p = self.segment().project(point)
		return EdgePos(self, p.distance(self.src.point))

	def is_opposite(self, edge):
		return edge.src == self.dst and edge.dst == self.src

	def get_opposite_edge(self):
		for edge in self.dst.out_edges:
			if self.is_opposite(edge):
				return edge
		return None

	def is_adjacent(self, edge):
		return edge.src == self.src or edge.src == self.dst or edge.dst == self.src or edge.dst == self.dst

	def orig_id(self):
		if hasattr(self, 'orig_edge_id'):
			return self.orig_edge_id
		else:
			return self.id

class EdgePos(object):
	def __init__(self, edge, distance):
		self.edge = edge
		self.distance = distance

	def point(self):
		segment = self.edge.segment()
		vector = segment.vector()
		if vector.magnitude() < 1:
			return segment.start
		else:
			return segment.start.add(vector.scale(self.distance / vector.magnitude()))

	def reverse(self):
		return EdgePos(self.edge.get_opposite_edge(), self.edge.segment().length() - self.distance)

class Index(object):
	def __init__(self, graph, index):
		self.graph = graph
		self.index = index

	def search(self, rect):
		edge_ids = self.index.intersection((rect.start.x, rect.start.y, rect.end.x, rect.end.y))
		return [self.graph.edges[edge_id] for edge_id in edge_ids]

	def subgraph(self, rect):
		return graph_from_edges(self.search(rect))

def graph_from_edges(edges):
	ng = Graph()
	vertex_map = {}
	for edge in edges:
		for vertex in [edge.src, edge.dst]:
			if vertex not in vertex_map:
				vertex_map[vertex] = ng.add_vertex(vertex.point)
		nedge = ng.add_edge(vertex_map[edge.src], vertex_map[edge.dst])
		nedge.orig_edge_id = edge.orig_id()
	return ng

class Graph(object):
	def __init__(self):
		self.vertices = []
		self.edges = []

	def add_vertex(self, point):
		vertex = Vertex(len(self.vertices), point)
		self.vertices.append(vertex)
		return vertex

	def find_edge(self, src, dst):
		for edge in src.out_edges:
			if edge.dst == dst:
				return edge
		return None

	def add_edge(self, src, dst):
		if src == dst:
			raise Exception('cannot add edge between same vertex')
		elif self.find_edge(src, dst):
			return self.find_edge(src, dst)
		edge = Edge(len(self.edges), src, dst)
		self.edges.append(edge)
		src.out_edges.append(edge)
		dst.in_edges.append(edge)
		return edge

	def add_bidirectional_edge(self, src, dst):
		return (
			self.add_edge(src, dst),
			self.add_edge(dst, src),
		)

	def make_bidirectional(self):
		for edge in self.edges:
			self.add_edge(edge.dst, edge.src)

	def edge_grid_index(self, size):
		index = grid_index.GridIndex(size)
		for edge in self.edges:
			index.insert_rect(edge.bounds(), edge)
		return index

	def bounds(self):
		r = None
		for vertex in self.vertices:
			if r is None:
				r = geom.Rectangle(vertex.point, vertex.point)
			else:
				r = r.extend(vertex.point)
		return r

	def save(self, fname):
		with open(fname, 'w') as f:
			for vertex in self.vertices:
				f.write("{} {}\n".format(vertex.point.x, vertex.point.y))
			f.write("\n")
			for edge in self.edges:
				f.write("{} {}\n".format(edge.src.id, edge.dst.id))

	def clone(self):
		other = Graph()
		for vertex in self.vertices:
			v = other.add_vertex(vertex.point)
			if hasattr(vertex, 'edge_pos'):
				v.edge_pos = vertex.edge_pos
		for edge in self.edges:
			e = other.add_edge(other.vertices[edge.src.id], other.vertices[edge.dst.id])
		return other

	def filter_edges(self, filter_edges, keep_attrs=None):
		g = Graph()
		vertex_map = {}
		for edge in self.edges:
			if edge in filter_edges:
				continue
			for vertex in [edge.src, edge.dst]:
				if vertex not in vertex_map:
					vertex_map[vertex] = g.add_vertex(vertex.point)
			new_edge = g.add_edge(vertex_map[edge.src], vertex_map[edge.dst])
			if keep_attrs:
				for k in keep_attrs:
					if hasattr(edge, k):
						setattr(new_edge, k, getattr(edge, k))
		return g

	def split_edge(self, edge, length):
		point = edge.segment().point_at_factor(length)
		new_vertex = self.add_vertex(point)

		orig_src, orig_dst = edge.src, edge.dst
		opp_edge = edge.get_opposite_edge()

		edge.dst = new_vertex
		orig_dst.in_edges.remove(edge)
		new_vertex.in_edges.append(edge)
		remainder_edge = self.add_edge(new_vertex, orig_dst)
		remainder_edge.orig_edge_id = edge.orig_id()

		if opp_edge:
			opp_edge.src = new_vertex
			orig_dst.out_edges.remove(opp_edge)
			new_vertex.out_edges.append(opp_edge)
			e = self.add_edge(orig_dst, new_vertex)
			e.orig_edge_id = opp_edge.orig_id()

		return remainder_edge

	def union(self, other):
		g = self.clone()
		vertex_map = {}
		for edge in other.edges:
			for vertex in [edge.src, edge.dst]:
				if vertex not in vertex_map:
					vertex_map[vertex] = g.add_vertex(vertex.point)
			g.add_edge(vertex_map[edge.src], vertex_map[edge.dst])
		return g

	def closest_vertex(self, p):
		best_vertex = None
		best_distance = None
		for vertex in self.vertices:
			d = vertex.point.distance(p)
			if best_vertex is None or d < best_distance:
				best_vertex = vertex
				best_distance = d
		return best_vertex

def read_graph(fname, merge_duplicates=False, fpoint=False, verbose=True):
	point_obj = geom.Point
	if fpoint:
		point_obj = geom.FPoint

	graph = Graph()
	with open(fname, 'r') as f:
		vertex_section = True
		vertices = {}
		next_vertex_id = 0
		seen_points = {}
		for line in f:
			parts = line.strip().split(' ')
			if vertex_section:
				if len(parts) >= 2:
					point = point_obj(float(parts[0]), float(parts[1]))
					if point in seen_points and merge_duplicates:
						if verbose: print('merging duplicate vertex at {}'.format(point))
						vertices[next_vertex_id] = seen_points[point]
					else:
						vertex = graph.add_vertex(point)
						vertices[next_vertex_id] = vertex
						seen_points[point] = vertex
					next_vertex_id += 1
				else:
					vertex_section = False
			elif len(parts) >= 2:
				src = vertices[int(parts[0])]
				dst = vertices[int(parts[1])]
				if src == dst and merge_duplicates:
					if verbose: print('ignoring self edge at {}'.format(src.point))
					continue
				graph.add_edge(src, dst)
	return graph

def dijkstra_helper(src, stop_at=None, max_distance=None):
	distances = {}
	prev = {}
	remaining = set()
	seen = set()

	distances[src.id] = 0
	remaining.add(src)
	seen.add(src)

	while len(remaining) > 0:
		closestNode = None
		closestDistance = None
		for vertex in remaining:
			if closestNode is None or distances[vertex.id] < closestDistance:
				closestNode = vertex
				closestDistance = distances[vertex.id]
		remaining.remove(closestNode)

		if stop_at is not None and stop_at == closestNode:
			break
		elif closestDistance > max_distance:
			break

		for other, edge in closestNode._neighbors().items():
			if other not in seen:
				seen.add(other)
				remaining.add(other)
				distances[other.id] = float('inf')
				prev[other.id] = None

			if other in remaining:
				if hasattr(edge, 'cost'):
					d = closestDistance + edge.cost
				else:
					d = closestDistance + closestNode.point.distance(other.point)
				if d < distances[other.id]:
					distances[other.id] = d
					prev[other.id] = (closestNode, edge)

	return distances, prev

def shortest_distances_from_source(src, max_distance=None):
	distances, _ = dijkstra_helper(src, max_distance=max_distance)
	return distances

def shortest_path(src, dst, max_distance=None):
	_, prev = dijkstra_helper(src, stop_at=dst, max_distance=max_distance)
	if dst.id not in prev:
		return None, None
	vertex_path = [dst]
	edge_path = []
	cur_node = dst
	while cur_node != src:
		cur_node, edge = prev[cur_node.id]
		vertex_path.append(cur_node)
		edge_path.append(edge)
	vertex_path.reverse()
	edge_path.reverse()
	return vertex_path, edge_path

# searches for the closest edge position to the specified point
# if src is specified, only looks for edges reachable from the src edge position within remaining units
def closest_reachable_edge(point, index, explored_node_pairs=None, remaining = 50, src = None, distance_threshold = 50):
	closest_edge_pos = None
	closest_edge_pos_path = None
	smallest_distance = None

	# if src is None, get candidate edges from index
	# otherwise, get candidate edges by following graph
	if src is None:
		candidates = [(edge, None) for edge in index.search(point.bounds().add_tol(distance_threshold))]
	else:
		candidates = [(src.edge, None)]
		cur_explored_node_pairs = set()
		cur_explored_node_pairs.add((src.edge.src.id, src.edge.dst.id))
		def search_edge(path, edge, remaining):
			l = edge.segment().length()
			if remaining > l:
				search_vertex(path + [edge], edge.dst, remaining - l)
		def search_vertex(path, vertex, remaining):
			for edge in vertex.out_edges:
				if (edge.src.id, edge.dst.id) in cur_explored_node_pairs or (edge.dst.id, edge.src.id) in cur_explored_node_pairs:
					continue
				candidates.append((edge, path))
				cur_explored_node_pairs.add((edge.src.id, edge.dst.id))
				search_edge(path, edge, remaining)

		# search forwards
		l = src.edge.segment().length() - src.distance
		if remaining > l:
			search_vertex([], src.edge.dst, remaining - l)

		# search backwards
		reverse_edge_pos = src.reverse()
		l = reverse_edge_pos.edge.segment().length() - reverse_edge_pos.distance
		if remaining > l:
			search_vertex([], reverse_edge_pos.edge.dst, remaining - l)

	for edge, path in candidates:
		if explored_node_pairs is not None and ((edge.src.id, edge.dst.id) in explored_node_pairs or (edge.dst.id, edge.src.id) in explored_node_pairs) and (src is None or edge != src.edge):
			continue
		distance = edge.segment().distance(point)
		if distance > distance_threshold:
			continue

		if closest_edge_pos is None or distance < smallest_distance:
			closest_edge_pos = edge.closest_pos(point)
			closest_edge_pos_path = path
			smallest_distance = distance

	return closest_edge_pos, closest_edge_pos_path

def follow_graph(edge_pos, distance, explored_node_pairs=None):
	if explored_node_pairs:
		explored_node_pairs = set(explored_node_pairs)
	else:
		explored_node_pairs = set()
	explored_node_pairs.add((edge_pos.edge.src.id, edge_pos.edge.dst.id))
	positions = []

	def search_edge(edge, remaining):
		l = edge.segment().length()
		if remaining > l:
			search_vertex(edge.dst, remaining - l)
		else:
			pos = EdgePos(edge, remaining)
			positions.append(pos)

	def search_vertex(vertex, remaining):
		for edge in vertex.out_edges:
			if (edge.src.id, edge.dst.id) in explored_node_pairs or (edge.dst.id, edge.src.id) in explored_node_pairs:
				continue
			explored_node_pairs.add((edge.src.id, edge.dst.id))
			search_edge(edge, remaining)

	remaining = distance
	l = edge_pos.edge.segment().length() - edge_pos.distance
	if remaining > l:
		search_vertex(edge_pos.edge.dst, remaining - l)
	else:
		positions = [EdgePos(edge_pos.edge, edge_pos.distance + remaining)]

	return positions

class RoadSegment(object):
	def __init__(self, id):
		self.id = id
		self.edges = []

	def add_edge(self, edge, direction):
		if direction == 'forwards':
			self.edges.append(edge)
		elif direction == 'backwards':
			self.edges = [edge] + self.edges
		else:
			raise Exception('bad edge')

	def compute_edge_distances(self):
		l = 0
		self.edge_distances = {}
		for edge in self.edges:
			self.edge_distances[edge.id] = l
			l += edge.segment().length()

	def distance_to_edge(self, distance, return_idx=False):
		for i in xrange(len(self.edges)):
			edge = self.edges[i]
			distance -= edge.segment().length()
			if distance <= 0:
				if return_idx:
					return i
				else:
					return edge
		if return_idx:
			return len(self.edges) - 1
		else:
			return self.edges[-1]

	def src(self):
		return self.edges[0].src

	def dst(self):
		return self.edges[-1].dst

	def is_opposite(self, rs):
		return self.src() == rs.dst() and self.dst() == rs.src() and self.edges[0].is_opposite(rs.edges[-1])

	def in_rs(self, edge_to_rs):
		rs_set = {}
		for edge in self.src().in_edges:
			rs = edge_to_rs[edge.id]
			if rs.id != self.id and rs.id not in rs_set:
				rs_set[rs.id] = rs
		return rs_set.values()

	def out_rs(self, edge_to_rs):
		rs_set = {}
		for edge in self.dst().out_edges:
			rs = edge_to_rs[edge.id]
			if rs.id != self.id and rs.id not in rs_set:
				rs_set[rs.id] = rs
		return rs_set.values()

	def get_opposite_rs(self, edge_to_rs):
		for rs in self.out_rs(edge_to_rs):
			if self.is_opposite(rs):
				return rs
		return None

	def length(self):
		return sum([edge.segment().length() for edge in self.edges])

	def closest_pos(self, point):
		best_edge_pos = None
		best_distance = None
		for edge in self.edges:
			edge_pos = edge.closest_pos(point)
			distance = edge_pos.point().distance(point)
			if best_edge_pos is None or distance < best_distance:
				best_edge_pos = edge_pos
				best_distance = distance
		return best_edge_pos

	def point_at_factor(self, t):
		edge = self.distance_to_edge(t)
		return edge.segment().point_at_factor(t - self.edge_distances[edge.id])

	def bounds(self):
		r = None
		for edge in self.edges:
			if r is None:
				r = edge.bounds()
			else:
				r = r.extend_rect(edge.bounds())
		return r

def get_graph_road_segments(g):
	road_segments = []
	edge_to_rs = {}

	def search_from_edge(rs, edge, direction):
		cur_edge = edge
		while True:
			if direction == 'forwards':
				vertex = cur_edge.dst
				edges = vertex.out_edges
			elif direction == 'backwards':
				vertex = cur_edge.src
				edges = vertex.in_edges

			edges = [next_edge for next_edge in edges if not next_edge.is_opposite(cur_edge)]

			if len(edges) != 1:
				# we have hit intersection vertex or a dead end
				return

			next_edge = edges[0]

			if next_edge.id in edge_to_rs:
				# this should only happen when we run in a segment that is actually a loop
				# although really it shouldn't happen in that case either, since loops should start/end at an intersection
				# TODO: think about this more
				return

			rs.add_edge(next_edge, direction)
			edge_to_rs[next_edge.id] = rs
			cur_edge = next_edge

	for edge in g.edges:
		if edge.id in edge_to_rs:
			continue

		rs = RoadSegment(len(road_segments))
		rs.add_edge(edge, 'forwards')
		edge_to_rs[edge.id] = rs
		search_from_edge(rs, edge, 'forwards')
		search_from_edge(rs, edge, 'backwards')
		rs.compute_edge_distances()
		road_segments.append(rs)

	return road_segments, edge_to_rs

class GraphContainer(object):
	def __init__(self, g):
		self.graph = g
		self.edge_index = g.edgeIndex()
		self.road_segments, self.edge_to_rs = get_graph_road_segments(g)

def mapmatch(index, road_segments, edge_to_rs, points, segment_length):
	SIGMA = segment_length
	START_TOL = segment_length * 2.5
	MAX_TOL = segment_length * 4

	# probs keeps track of both the emission probability and the distance along road segment of the
	#  point closest to the previous path point
	# on the next iteration, we restrict to choosing points at higher distances along road segment
	probs = {}

	for edge in index.search(points[0].bounds().add_tol(START_TOL)):
		rs = edge_to_rs[edge.id]
		distance = edge.segment().distance(points[0])
		p = -0.5 * distance * distance / SIGMA / SIGMA
		if rs.id not in probs or p > probs[rs.id][0]:
			rs_distance = rs.edge_distances[edge.id] + edge.segment().project_factor(points[0])
			probs[rs.id] = (p, rs_distance)

	# given a road segment and a previous rs distance, returns next rs distance and the actual point distance
	def distance_to_rs(rs, clip_distance, point, vector):
		low_distance = clip_distance + segment_length / 2
		high_distance = clip_distance + segment_length * 2

		if high_distance < 0 or low_distance > rs.length():
			return None, None

		# get edges between low_distance and high_distance
		low_edge_idx = rs.distance_to_edge(low_distance, return_idx=True)
		high_edge_idx = rs.distance_to_edge(high_distance, return_idx=True)
		edges = rs.edges[low_edge_idx:high_edge_idx+1]

		best_distance = None
		best_rs_distance = None
		for edge in edges:
			rs_distance = rs.edge_distances[edge.id] + edge.segment().project_factor(point)
			rs_distance = numpy.clip(rs_distance, low_distance, high_distance)
			rs_distance = numpy.clip(rs_distance, rs.edge_distances[edge.id], rs.edge_distances[edge.id] + edge.segment().length())
			edge_distance = rs_distance - rs.edge_distances[edge.id]
			rs_point = edge.segment().point_at_factor(edge_distance)
			distance = rs_point.distance(point) + edge.segment().vector().angle_to(vector) * 12
			if best_distance is None or distance < best_distance:
				best_distance = distance
				best_rs_distance = rs_distance

		return best_distance, best_rs_distance

	backpointers = [{} for _ in xrange(len(points) - 1)]

	for i in xrange(len(points) - 1):
		next_probs = {}
		for prev_rs_id in probs:
			prev_p, prev_rs_distance = probs[prev_rs_id]
			rs = road_segments[prev_rs_id]
			remaining_rs_distance = rs.length() - prev_rs_distance
			rs_candidates = [(rs, prev_rs_distance)] + [(next_rs, -remaining_rs_distance) for next_rs in rs.out_rs(edge_to_rs)]
			for next_rs, clip_distance in rs_candidates:
				if next_rs.is_opposite(rs):
					continue
				distance, rs_distance = distance_to_rs(next_rs, clip_distance, points[i + 1], points[i + 1].sub(points[i]))
				if distance is None or distance > MAX_TOL:
					continue
				p = prev_p + (-0.5 * distance * distance / SIGMA / SIGMA)
				#if next_rs != rs:
				#	p += math.log(0.1)
				if next_rs.id not in next_probs or p > next_probs[next_rs.id][0]:
					next_probs[next_rs.id] = (p, rs_distance)
					backpointers[i][next_rs.id] = prev_rs_id
		probs = next_probs
		if len(probs) == 0:
			return None, None

	return probs, backpointers

def mm_best_rs(road_segments, probs, rs_blacklist=None):
	best_rs = None
	for rs_id in probs:
		if rs_blacklist is not None and rs_id in rs_blacklist:
			continue
		prob = probs[rs_id][0]
		if best_rs is None or prob > probs[best_rs.id][0]:
			best_rs = road_segments[rs_id]
	return best_rs

def mm_follow_backpointers(road_segments, rs_id, backpointers):
	rs_list = []
	for i in xrange(len(backpointers) - 1, -1, -1):
		rs_id = backpointers[i][rs_id]
		rs_list.append(road_segments[rs_id])
	rs_list.reverse()
	return rs_list

def get_nearby_vertices(vertex, n):
	nearby_vertices = set()
	def search(vertex, remaining):
		if vertex in nearby_vertices:
			return
		nearby_vertices.add(vertex)
		if remaining == 0:
			return
		for edge in vertex.in_edges:
			search(edge.src, remaining - 1)
			search(edge.dst, remaining - 1)
	search(vertex, n)
	return nearby_vertices

def get_nearby_vertices_by_distance(vertex, distance):
	nearby_vertices = set([vertex])
	search_queue = []
	search_queue.append((vertex, distance))
	while len(search_queue) > 0:
		vertex, remaining = search_queue[0]
		search_queue = search_queue[1:]
		if remaining <= 0:
			continue
		for edge in vertex.out_edges:
			if edge.dst in nearby_vertices:
				continue
			nearby_vertices.add(edge.dst)
			search_queue.append((edge.dst, remaining - edge.segment().length()))
	return nearby_vertices

def densify(g, length, epsilon=0.1):
	for edge in g.edges:
		n_split = int(edge.segment().length() / length - epsilon)
		for i in xrange(n_split):
			edge = g.split_edge(edge, length)
