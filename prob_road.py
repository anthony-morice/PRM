import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.neighbors import NearestNeighbors
from queue import Queue

class PRM:
  def __init__(self, cspace, n=1000, k=5, rrt_flag=False):
    self.cspace = self.read_image(cspace)
    self.build_roadmap(n, k, rrt=rrt_flag)

  def read_image(self, path):
    #im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    #return cv2.threshold(im, 254, 255, cv2.THRESH_BINARY)[1]
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

  def view_cspace(self):
    plt.imshow(self.cspace, cmap='gray')
    plt.show()

  def obstructed(self, i, j):
    return self.cspace[i,j] != 255

  def valid_path(self, a, b):
    # bresenhams line generation algorithm
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    flag = dx <= dy
    if flag:
      dx, dy = dy, dx
      a = (a[1], a[0])
      b = (b[1], b[0])
    ax, ay = a
    bx, by = b
    residual = 2 * dy - dx
    for _ in range(dx):
      ax = ax + 1 if ax < bx else ax - 1
      if residual >= 0:
        ay = ay + 1 if ay < by else ay - 1
        residual -= 2 * dx
      residual += 2 * dy
      if flag:
        if self.obstructed(ay, ax):
          return False
      else:
        if self.obstructed(ax, ay):
          return False
    return True

  def roadmap_classic(self, n, k):
    # generate random samples in configuration space
    samples = []
    count = 0
    while (len(samples) < n and count < 2 * n):
      point = (np.random.randint(0, self.cspace.shape[0]),\
               np.random.randint(0, self.cspace.shape[1]))
      if not self.obstructed(point[0], point[1]): # discard obstructed points
        samples.append(point)
      count += 1
    if count == 2 * n:
      print(f'Note: only {len(samples)} out of {n} vertices were generated for roadmap')
    self.prm_vertices = np.array(samples)
    # attempt to connect each sample to n-nearest neighbors
    self.knn = NearestNeighbors(n_neighbors=k+1)
    self.knn.fit(self.prm_vertices) # fit to samples
    indices = self.knn.kneighbors(self.prm_vertices, return_distance=False)
    self.prm_edges = [[] for _ in range(self.prm_vertices.shape[0])]
    for i in range(indices.shape[0]):
      for nn in indices[i,1:]:
        if self.valid_path(self.prm_vertices[i], self.prm_vertices[nn]) and nn not in self.prm_edges[i]:
          self.prm_edges[i].append(nn)
          self.prm_edges[nn].append(i)
    # enhancement phase? disjoint sets TODO

  def euclid_dist(a, b):
    return np.linalg.norm(np.array([a[i] - b[i] for i in range(2)]))

  def nearest(samples, point):
    assert len(samples) > 0
    distances = np.linalg.norm(np.array(samples) - point, axis=1)
    return np.argmin(distances)

  def roadmap_rrt(self, n, k, step=0.5):
    # rapidly-exploring random trees
    # add tree root
    samples = []
    count = 0
    self.prm_edges = [[]]
    while count < 100:
      point = (np.random.randint(0, self.cspace.shape[0]),\
                np.random.randint(0, self.cspace.shape[1]))
      if not self.obstructed(point[0], point[1]): # only accept valid point 
        samples.append(np.array(point))
        break
      count += 1
    if count == 100:
      print('Error: unable to generate unobstructed tree root for rrt')
      exit(1)
    count = 0
    while (len(samples) < n and count < 2 * n):
      point = np.array((np.random.randint(0, self.cspace.shape[0]),\
               np.random.randint(0, self.cspace.shape[1])))
      if not self.obstructed(point[0], point[1]):
        # find nearest existing vertex to generated point
        vertex_near = PRM.nearest(samples, point)
        # if path not obstructed, take small step towards point and create edge + vertex
        if self.valid_path(samples[vertex_near], point):
          samples.append((samples[vertex_near] + (point - samples[vertex_near]) * step).astype(int))
          self.prm_edges.append([])
          self.prm_edges[-1].append(vertex_near)
          self.prm_edges[vertex_near].append(len(self.prm_edges) - 1)
      count += 1
    if count == 2 * n:
      print(f'Note: only {len(samples)} out of {n} vertices were generated for roadmap')
    self.prm_vertices = np.array(samples)

  def build_roadmap(self, n=1000, k=7, rrt=False):
    if rrt:
      self.roadmap_rrt(n,k, step=0.5)
      self.knn = NearestNeighbors(n_neighbors=k+1)
      self.knn.fit(self.prm_vertices) # fit to samples
    else:
      self.roadmap_classic(n, k)

  def view_roadmap(self):
    _, ax = plt.subplots()
    # plot cspace
    plt.imshow(self.cspace, cmap='gray')
    # plot edges
    lines = []
    for i in range(self.prm_vertices.shape[0]):
      for edge in self.prm_edges[i]:
        lines.append([(self.prm_vertices[i,1],self.prm_vertices[i,0]),\
                      (self.prm_vertices[edge,1],self.prm_vertices[edge,0])])
    lc = LineCollection(lines, color='c', linewidths=1)
    ax.add_collection(lc)
    # plot vertices
    plt.plot(self.prm_vertices[:,1], self.prm_vertices[:,0], 'ro', ms=3) 
    plt.show()

  def bfs(self, start, goal):
    seen = {start}
    q = Queue()
    q.put((start, [start]))
    while not q.empty():
      vertex, path = q.get()
      if vertex == goal:
        path.append(goal)
        return True, path
      for edge in self.prm_edges[vertex]:
        if edge not in seen:
          seen.add(edge)
          q.put((edge, path.copy() + [edge]))
    return False, []

  def get_path(self, start, end):
    if self.obstructed(start[0], start[1]) or self.obstructed(end[0], end[1]):
      print('Error: start or end points are invalid')
      return False, []
    # connect start and end to prm with local search
    indices = self.knn.kneighbors([start,end], return_distance=False)
    points = [start, end]
    in_network = []
    for i in range(2):
      for candidate in indices[i]:
        if self.valid_path(points[i], self.prm_vertices[candidate]):
          in_network.append(candidate)
          break
    if len(in_network) != 2:
      print('Error: could not connect start and end points to roadmap')
      return False, []
    # path plan 
    res, path = self.bfs(in_network[0], in_network[1])
    if res is False:
      print('Error: BFS failed')
      return False, []
    # refine path? TODO
    for i, vertex in enumerate(path):
      path[i] = self.prm_vertices[vertex]
    path = [start] + path + [end]
    return True, path

  def view_path(self, path):
    _, ax = plt.subplots()
    # plot cspace
    plt.imshow(self.cspace, cmap='gray')
    # plot edges
    lines = []
    for i in range(self.prm_vertices.shape[0]):
      for edge in self.prm_edges[i]:
        lines.append([(self.prm_vertices[i,1],self.prm_vertices[i,0]),\
                      (self.prm_vertices[edge,1],self.prm_vertices[edge,0])])
    lc = LineCollection(lines, color='c', linewidths=1)
    ax.add_collection(lc)
    # plot path
    lines = []
    for i in range(len(path) - 1):
      lines.append([(path[i][1], path[i][0]), (path[i+1][1], path[i+1][0])])
    lc = LineCollection(lines, color='g', linewidths=4)
    ax.add_collection(lc)
    # plot vertices
    plt.plot(self.prm_vertices[:,1], self.prm_vertices[:,0], 'ro', ms=3) 
    # plot start and goal points
    plt.plot([path[0][1], path[-1][1]],[path[0][0], path[-1][0]] , 'mo', ms=8) 
    plt.show()

def main():
  prm = PRM("./images/test.png", n=1000, rrt_flag=True)
  res, path = prm.get_path((10, 10), (342, 642))
  if res:
    prm.view_path(path)

if __name__ == "__main__":
  main()
