"""
Autonomous Delivery Agent - Single-file reference implementation

Contents:
- Grid and Environment (static obstacles, terrain costs, dynamic moving obstacles with schedule)
- Planners: BFS, Uniform Cost Search (UCS), A* (Manhattan heuristic), Simulated Annealing replanner (local search)
- Experiment runner: runs planners on 4 built-in maps (small, medium, large, dynamic)
- CLI for running single trials, experiments, and producing logs for dynamic replanning
- README and short report (markdown) embedded at bottom. Save this file and run with Python 3.8+

Usage examples:
  python autonomous_delivery_agent.py --list-maps
  python autonomous_delivery_agent.py --map small --planner astar
  python autonomous_delivery_agent.py --experiment all --repeats 3
  python autonomous_delivery_agent.py --map dynamic --planner ucs --simulate-dynamic

Notes:
- This is a proof-of-concept teaching implementation. It focuses on clarity and reproducibility.
- The dynamic replanning demo shows an obstacle appearing during execution and the agent invoking replanning (A* or SA) when the next step is blocked.

"""

import argparse
import time
import heapq
import random
import copy
import math
import sys
from collections import deque, namedtuple, defaultdict

# ----------------------------- Data structures -----------------------------

Point = namedtuple('Point', ['r', 'c'])

class Grid:
    """Grid holds integer movement costs >=1, and '#' for impassable cells."""
    def __init__(self, grid_costs):
        # grid_costs: list of lists; each entry either int>=1 or '#' for obstacle
        self.grid = grid_costs
        self.R = len(grid_costs)
        self.C = len(grid_costs[0]) if self.R>0 else 0

    def in_bounds(self, p: Point):
        return 0 <= p.r < self.R and 0 <= p.c < self.C

    def passable(self, p: Point):
        return self.in_bounds(p) and self.grid[p.r][p.c] != '#'

    def cost(self, p: Point):
        if not self.in_bounds(p):
            return math.inf
        v = self.grid[p.r][p.c]
        if v == '#':
            return math.inf
        return int(v)

    def neighbors(self, p: Point):
        # 4-connected movement
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            np = Point(p.r+dr, p.c+dc)
            if self.in_bounds(np):
                yield np

    def pretty(self, path=None, start=None, goal=None):
        grid = [[str(self.grid[r][c]) for c in range(self.C)] for r in range(self.R)]
        if path:
            for p in path:
                grid[p.r][p.c] = '*'
        if start:
            grid[start.r][start.c] = 'S'
        if goal:
            grid[goal.r][goal.c] = 'G'
        return '\n'.join(''.join(cell).replace('#','#') for cell in grid)

# ----------------------------- Environment ----------------------------------

class DynamicObstacle:
    """Moving obstacle with deterministic schedule: a dict time->Point or list of (t,Point)
    For simplicity, schedule is a dict mapping integer timestep to cell occupied."""
    def __init__(self, schedule):
        # schedule: dict int->Point
        self.schedule = schedule

    def occupies(self, t):
        return self.schedule.get(t, None)

class Environment:
    def __init__(self, base_grid: Grid, dynamic_obstacles=None):
        self.base = base_grid
        self.dynamic = dynamic_obstacles if dynamic_obstacles else []

    def is_cell_blocked_at(self, p: Point, t:int):
        # blocked if base is obstacle or any dynamic occupies p at time t
        if not self.base.in_bounds(p):
            return True
        if self.base.grid[p.r][p.c] == '#':
            return True
        for dob in self.dynamic:
            occ = dob.occupies(t)
            if occ and occ == p:
                return True
        return False

    def cost_at(self, p: Point, t:int):
        # returns cost if passable at time t, else math.inf
        if self.is_cell_blocked_at(p,t):
            return math.inf
        return self.base.cost(p)

# ----------------------------- Planners ------------------------------------

class SearchResult:
    def __init__(self, path, cost, nodes_expanded, time_sec):
        self.path = path
        self.cost = cost
        self.nodes_expanded = nodes_expanded
        self.time = time_sec

def reconstruct(parent, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur, None)
    path.reverse()
    return path

# BFS (on unweighted grid treating cost as uniform 1): returns shortest in steps
def bfs(env: Environment, start: Point, goal: Point, max_time=10.0):
    t0 = time.time()
    frontier = deque([start])
    parent = {start: None}
    visited = set([start])
    nodes = 0
    while frontier:
        if time.time()-t0 > max_time:
            break
        cur = frontier.popleft()
        nodes += 1
        if cur == goal:
            path = reconstruct(parent, goal)
            cost = len(path)-1
            return SearchResult(path, cost, nodes, time.time()-t0)
        for n in env.base.neighbors(cur):
            if n not in visited and not env.base.grid[n.r][n.c]=='#':
                visited.add(n)
                parent[n]=cur
                frontier.append(n)
    return SearchResult(None, math.inf, nodes, time.time()-t0)

# Uniform Cost Search (Dijkstra)
def ucs(env: Environment, start: Point, goal: Point, time_limit=10.0, consider_time=False):
    t0=time.time()
    pq=[]
    heapq.heappush(pq, (0, start))
    parent={start: None}
    cost_so_far={start: 0}
    nodes=0
    while pq:
        if time.time()-t0>time_limit:
            break
        c, cur = heapq.heappop(pq)
        nodes+=1
        if cur==goal:
            return SearchResult(reconstruct(parent, goal), c, nodes, time.time()-t0)
        for n in env.base.neighbors(cur):
            if env.base.grid[n.r][n.c]=='#':
                continue
            new_cost = c + env.base.cost(n)
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n]=new_cost
                parent[n]=cur
                heapq.heappush(pq, (new_cost, n))
    return SearchResult(None, math.inf, nodes, time.time()-t0)

# A* with Manhattan heuristic (admissible)
def manhattan(a:Point, b:Point):
    return abs(a.r-b.r)+abs(a.c-b.c)

def astar(env: Environment, start: Point, goal: Point, time_limit=10.0):
    t0=time.time()
    open_pq=[]
    heapq.heappush(open_pq, (manhattan(start,goal), 0, start))
    parent={start: None}
    g={start:0}
    nodes=0
    while open_pq:
        if time.time()-t0>time_limit:
            break
        _, cost_so_far, cur = heapq.heappop(open_pq)
        nodes+=1
        if cur==goal:
            return SearchResult(reconstruct(parent, goal), g[goal], nodes, time.time()-t0)
        for n in env.base.neighbors(cur):
            if env.base.grid[n.r][n.c]=='#':
                continue
            tentative = g[cur] + env.base.cost(n)
            if n not in g or tentative < g[n]:
                g[n]=tentative
                parent[n]=cur
                f = tentative + manhattan(n,goal)
                heapq.heappush(open_pq, (f, tentative, n))
    return SearchResult(None, math.inf, nodes, time.time()-t0)

# ------------------ Local search replanner (Simulated Annealing) -------------

def path_cost(env: Environment, path):
    if path is None:
        return math.inf
    cost=0
    for p in path[1:]:
        if env.base.grid[p.r][p.c]=='#':
            return math.inf
        cost += env.base.cost(p)
    return cost

def simulated_annealing_replan(env: Environment, current_path, start_index, goal, time_limit=2.0):
    """Given a current_path (list of Points), attempt to improve it using SA
    start_index: index in path where agent currently is (we only replan the suffix)
    Strategy: pick two indices i<j in suffix, try to replace path[i:j] with A* between endpoints
    Accept changes probabilistically.
    """
    t0=time.time()
    best = current_path[:] if current_path else None
    best_cost = path_cost(env, best)
    if best is None:
        return None
    T0 = 1.0
    it=0
    while time.time()-t0 < time_limit:
        it+=1
        suffix = best[start_index:]
        if len(suffix) < 3:
            break
        i = start_index + random.randint(0, max(0, len(best)-start_index-2))
        j = i + 1 + random.randint(0, max(0, len(best)-i-1))
        a = best[i]
        b = best[j]
        # run A* between a and b on same static grid
        sub = astar(env, a, b, time_limit=0.5)
        if sub.path is None:
            continue
        new_path = best[:i] + sub.path + best[j+1:]
        new_cost = path_cost(env, new_path)
        if new_cost < best_cost:
            best = new_path
            best_cost = new_cost
        else:
            # acceptance probability
            T = T0 * (0.99 ** it)
            if T <= 1e-6:
                break
            p_accept = math.exp((best_cost - new_cost)/T)
            if random.random() < p_accept:
                best = new_path
                best_cost = new_cost
    return best

# ----------------------------- Agent Execution -------------------------------

class Agent:
    def __init__(self, env: Environment, start: Point, goal: Point, planner='astar', replanner='sa'):
        self.env = env
        self.start = start
        self.goal = goal
        self.planner = planner
        self.replanner = replanner
        self.path = None
        self.time = 0 # timestep during execution
        self.log = []

    def plan(self):
        if self.planner=='bfs':
            res = bfs(self.env, self.start, self.goal)
        elif self.planner=='ucs':
            res = ucs(self.env, self.start, self.goal)
        else:
            res = astar(self.env, self.start, self.goal)
        self.path = res.path
        self.log.append((self.time, 'initial_plan', res.cost, res.nodes_expanded))
        return res

    def step_execute(self, max_steps=1000, dynamic_replan=True):
        """Execute following planned path, step by step. If next cell becomes blocked at time t+1, trigger replanning."""
        if not self.path:
            return {'success':False, 'reason':'no_path', 'log':self.log}
        cur_idx = 0
        # find index of current position in path (should be 0)
        total_cost = 0
        nodes_expanded = 0
        steps = 0
        while cur_idx < len(self.path)-1 and steps<max_steps:
            cur = self.path[cur_idx]
            nextp = self.path[cur_idx+1]
            # Check if next cell is blocked at time+1
            if self.env.is_cell_blocked_at(nextp, self.time+1):
                self.log.append((self.time+1, 'blocked', nextp))
                if dynamic_replan:
                    # try replanning using chosen replanner
                    self.log.append((self.time+1, 'replanning_start'))
                    if self.replanner=='sa':
                        new_path = simulated_annealing_replan(self.env, self.path, cur_idx, self.goal, time_limit=1.0)
                        method='simulated_annealing'
                    else:
                        # simply call A* from current position
                        res = astar(self.env, cur, self.goal)
                        new_path = res.path
                        method='astar'
                    self.log.append((self.time+1, 'replanning_end', method, len(new_path) if new_path else None))
                    if not new_path:
                        return {'success':False, 'reason':'no_replan_found', 'log':self.log}
                    # find new index mapping for agent (current stays at cur)
                    self.path = new_path
                    # ensure cur is at index 0
                    try:
                        cur_idx = self.path.index(cur)
                    except ValueError:
                        cur_idx = 0
                    continue
                else:
                    return {'success':False, 'reason':'blocked_no_replan', 'log':self.log}
            # move to next cell
            if self.env.is_cell_blocked_at(nextp, self.time+1):
                return {'success':False, 'reason':'blocked', 'log':self.log}
            move_cost = self.env.base.cost(nextp)
            total_cost += move_cost
            self.time += 1
            cur_idx += 1
            steps += 1
            nodes_expanded += 1
            self.log.append((self.time, 'moved', nextp, move_cost))
        success = (cur_idx==len(self.path)-1)
        return {'success':success, 'cost':total_cost, 'steps':steps, 'log':self.log}

# ----------------------------- Built-in maps --------------------------------

# We'll encode maps as lists of strings. Digit characters are terrain costs, '#' is wall.
MAPS = {}
MAPS['small'] = """
11111
1#111
111#1
1S11G
11111
"""
MAPS['medium'] = """
1111111111
1112122111
11##122211
1111222211
1122112211
11S111122G
1111111111
"""
MAPS['large'] = """
111111111111111
111122221112111
11##2222211221
11112222221111
11222111222211
11112221112211
11S111111122G1
111111111111111
"""
# dynamic map: there will be a moving vehicle crossing the shortest path at timestep=3
MAPS['dynamic'] = """
111111
1S1111
111#11
111111
11#1G1
111111
"""

def parse_map(text):
    lines = [ln for ln in text.strip().splitlines()]
    grid=[]
    start=None
    goal=None
    for r,ln in enumerate(lines):
        row=[]
        for c,ch in enumerate(ln.strip()):
            if ch=='S':
                start=Point(r,c)
                row.append('1')
            elif ch=='G':
                goal=Point(r,c)
                row.append('1')
            elif ch=='#':
                row.append('#')
            else:
                row.append(ch)
        grid.append(row)
    return Grid(grid), start, goal

# create dynamic obstacle schedule for dynamic map
def create_dynamic_obstacles_for_map(grid,start,goal):
    # Simple example: vehicle moves along a row crossing agent path.
    # We'll place a vehicle that occupies (2,2) at t=2, (3,2) at t=3, etc.
    schedule = {}
    # pick a straight line between two points
    for t in range(2,8):
        schedule[t]=Point(2+(t-2),2) if 2+(t-2) < grid.R else None
    dob = DynamicObstacle(schedule)
    return [dob]

# ----------------------------- Experiments ---------------------------------

def run_trial(mapname, planner='astar', replanner='sa', simulate_dynamic=False, verbose=False):
    grid, start, goal = parse_map(MAPS[mapname])
    dynamic = None
    if mapname=='dynamic' or simulate_dynamic:
        dynamic = create_dynamic_obstacles_for_map(grid,start,goal)
    env = Environment(grid, dynamic_obstacles=dynamic)
    agent = Agent(env, start, goal, planner=planner, replanner=replanner)
    plan_res = agent.plan()
    exec_res = agent.step_execute()
    return {'map':mapname, 'planner':planner, 'replanner':replanner, 'plan_cost':plan_res.cost, 'plan_nodes':plan_res.nodes_expanded, 'plan_time':plan_res.time, 'exec':exec_res}

def run_experiments(repeats=3):
    planners = ['bfs','ucs','astar']
    maps = ['small','medium','large','dynamic']
    rows = []
    for m in maps:
        for p in planners:
            for r in range(repeats):
                res = run_trial(m, planner=p, replanner='sa', simulate_dynamic=(m=='dynamic'))
                rows.append(res)
    return rows

# ----------------------------- CLI -----------------------------------------

README = """
Autonomous Delivery Agent - README

Requirements: Python 3.8+

Run single map:
  python autonomous_delivery_agent.py --map small --planner astar

Run experiments (default 3 repeats):
  python autonomous_delivery_agent.py --experiment all --repeats 3

List available maps:
  python autonomous_delivery_agent.py --list-maps

"""

SHORT_REPORT = """
# Short report (1-2 pages) - Environment & Agent (expand for full deliverable)

Environment model:
- Grid world with integer movement costs >=1. Obstacles marked as '#'.
- Dynamic obstacles modeled by deterministic schedule mapping timestep->cell.
- Agent executes one move per timestep; dynamic obstacles can block cells at certain timesteps.

Agent design:
- Plans using BFS (unweighted), Uniform-Cost Search (Dijkstra), or A* with Manhattan heuristic (admissible).
- Replanning: triggered when next step is blocked; options: immediate A* replanning or simulated annealing local search improving current path.

Heuristics:
- Manhattan distance for A* (admissible in 4-connected grid with unit step cost; still admissible when costs>=1 because it underestimates cost).

Experimental results:
- The provided experiment runner will run each planner on 4 maps (small, medium, large, dynamic) and record plan cost, nodes expanded, and runtime.

Analysis and conclusion:
- BFS fastest on small uniform-cost maps when costs are effectively uniform; fails to account for weights.
- UCS finds optimal-cost path when costs vary but expands more nodes compared to A*.
- A* significantly reduces nodes expanded when heuristic is informative.
- Simulated annealing can help find alternative paths without full re-search when dynamic changes are local; but may not guarantee optimality.

"""

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list-maps', action='store_true')
    parser.add_argument('--map', type=str, help='map name (small, medium, large, dynamic)')
    parser.add_argument('--planner', type=str, default='astar', help='bfs|ucs|astar')
    parser.add_argument('--replanner', type=str, default='sa', help='sa|astar')
    parser.add_argument('--experiment', type=str, help='all to run full experiment')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--simulate-dynamic', action='store_true')
    parser.add_argument('--show-report', action='store_true')
    args = parser.parse_args()
    if args.list_maps:
        print('Available maps:', ', '.join(MAPS.keys()))
        sys.exit(0)
    if args.show_report:
        print(SHORT_REPORT)
        sys.exit(0)
    if args.experiment:
        if args.experiment=='all':
            rows = run_experiments(repeats=args.repeats)
            print('Experiment results (sample):')
            for r in rows:
                print(r)
            sys.exit(0)
    if args.map:
        if args.map not in MAPS:
            print('Unknown map. Use --list-maps')
            sys.exit(1)
        res = run_trial(args.map, planner=args.planner, replanner=args.replanner, simulate_dynamic=args.simulate_dynamic)
        print('Result:')
        print(res)
        print('\nMap layout:')
        g, s, gg = parse_map(MAPS[args.map])
        print(g.pretty(start=s, goal=gg))
        sys.exit(0)
    print(README)
