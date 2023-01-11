import math

class GridIndex(object):
	def __init__(self, size):
		self.size = size
		self.grid = {}

	# Insert a point with data.
	def insert(self, p, data):
		self.insert_rect(p.bounds(), data)

	# Insert a data with rectangle bounds.
	def insert_rect(self, rect, data):
		def f(cell):
			if cell not in self.grid:
				self.grid[cell] = []
			self.grid[cell].append(data)
		self.each_cell(rect, f)

	def each_cell(self, rect, f):
		for i in range(int(math.floor(rect.start.x / self.size)), int(math.floor(rect.end.x / self.size)) + 1):
			for j in range(int(math.floor(rect.start.y / self.size)), int(math.floor(rect.end.y / self.size)) + 1):
				f((i, j))

	def search(self, rect):
		matches = set()
		def f(cell):
			if cell not in self.grid:
				return
			for data in self.grid[cell]:
				matches.add(data)
		self.each_cell(rect, f)
		return matches
