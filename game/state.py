"""Board geometry (swappable map), game constants and the (mutable) game State.

The map is a 7x7 grid, rotationally symmetric so that neither seat has a
geometric advantage:

      c0 c1 c2 c3 c4 c5 c6
  r0   .  .  N  .  .  .  .        N = resource node (5 in total)
  r1   .  B0 .  .  N  .  .        B0/B1 = player bases
  r2   .  .  .  T  .  .  .        T = tree (impassable)
  r3   .  T  .  N  .  T  .
  r4   .  .  .  T  .  .  .
  r5   .  .  N  .  .  B1 .
  r6   .  .  .  .  N  .  .
"""
import random

SIZE = 7
N = SIZE * SIZE

# ---- rules constants -------------------------------------------------------
MAX_TURNS = 15          # turns per player (game = 2 * MAX_TURNS turns)
AP = 2                  # action points per turn
UNIT_HP = 3
MAX_UNITS = 4
BASE_HP = 5
START_RES = 3
COST_OUTPOST = 3
COST_RECRUIT = 4
COLLECT_DET = 2         # deterministic collect yield
SCORE_SCALE = 30.0      # tanh(diff / SCORE_SCALE) -> value in [-1, 1]
CENTER_STOCK = 9        # resource stockpile of the (contested) centre node
NODE_STOCK = 6          # stockpile of the other nodes
STOCK_REGEN = 1         # every node regrows this much at the end of each round
# score = resources + W_OUTPOST*outposts + W_UNIT*units + W_BASE*base_hp
W_OUTPOST, W_UNIT, W_BASE = 10, 8, 4

def index_of(r, c):
    #Convert row/column coordinates to a flat cell index.
    return r * SIZE + c

def coords_of_cell(c):
    #Convert a flat cell index back to row/column coordinates.
    return divmod(c, SIZE)

# ---- map (mutable in place so that `from game.state import NODES` stays valid) ----
DEFAULT_TREES = (cell(2, 3), cell(4, 3), cell(3, 1), cell(3, 5))
DEFAULT_NODES = (cell(3, 3), cell(1, 4), cell(5, 2), cell(0, 2), cell(6, 4))
DEFAULT_BASES = (cell(1, 1), cell(5, 5))

TREES = set()
NODES = []                 # NODES[0] is always the centre node (cell 24)
NODE_IDX = {}
BASES = [0, 0]
START_UNITS = [[], []]
NEIGH_DIR, NEIGH_SET, DIST, SPAWN = [], [], [], []

# directions: 0=up 1=right 2=down 3=left
DR = (-1, 0, 1, 0)
DC = (0, 1, 0, -1)
DIR_NAMES = ("up", "right", "down", "left")
CENTRE = cell(3, 3)

def rot(c):
    #180-degree rotation: maps player 0's side of the board onto player 1's.
    #Maps index `c` to `N - 1 - c`.
    return N - 1 - c


def _compute_map_structure(treelines, nodes_list, bases):
    """Computes neighbor lists, distance matrix, and valid spawn locations for both players.
    
    Args:
        treelines: Set of impassable tree cell indices.
        nodes_list: List of resource node cell indices (including centre).
        bases: List of two base cell indices [p0_base, p1_base].
    
    Returns:
        A tuple containing:
            - nd: 2D neighbor list for each cell (or -1 if out of bounds/into tree).
            - nset: Set of valid neighbor cells.
            - dist: Distance matrix between all non-tree cells.
            - spawn: List of candidate unit spawn positions per player.
    """
    nd = [[-1] * 4 for _ in range(N)]
    for r_idx in range(SIZE):
        for c_idx in range(SIZE):
            c_current = index_of(r_idx, c_idx)
            if c_current in treelines:
                continue
            
            for dir_idx in range(4):
                r_next, c_next = r_idx + DR[dir_idx], c_idx + DC[dir_idx]
                
                # Validate bounds and check for trees using the helper index function
                if 0 <= r_next < SIZE and 0 <= c_next < SIZE:
                    c_next_idx = index_of(r_next, c_next)
                    if c_next_idx not in treelines:
                        nd[c_current][dir_idx] = c_next_idx
    
    # Extract a flat frozenset of all valid (non-tree) cells from the neighbor matrix
    nset = [frozenset(cell for cell in row if cell >= 0) for row in nd]
    
    INF = 99
    dist = [[INF] * N for _ in range(N)]
    
    # Perform BFS to compute shortest path distances between all valid cells
    for start_cell in range(N):
        if start_cell in treelines:
            continue
            
        dist[start_cell][start_cell] = 0
        q = [start_cell]
        
        while q:
            u = q[0]
            del q[0]  # Pop front of queue
            
            for v_cell in nd[u]:
                if v_cell >= 0 and dist[start_cell][v_cell] == INF:
                    dist[start_cell][v_cell] = dist[start_cell][u] + 1
                    q.append(v_cell)
    
    spawn = []
    # Determine initial spawn candidates for each player
    for p in (0, 1):
        base = bases[p]
        # Start with the player's own base
        cand = [base]
        # Add neighbors of the base that are not trees and not the base itself
        neighbor_cells = nd[base]
        valid_neighbors = [x for x in neighbor_cells if x >= 0 and x != base]
        cand.extend(valid_neighbors)
        
        # Sort candidates: own base first, then by distance to the center node
        cand.sort(key=lambda c_idx: (c_idx != base, dist[c_idx][nodes_list[0]]))
        spawn.append(cand)
    
    return nd, nset, dist, spawn


def _apply_map(treelines, nodes_list, bases):
    """Applies a new map configuration to global geometry constants.
    
    Updates the global `TREES`, `NODES`, and `NEIGH_DIR` lists/sets, as well as
    precomputing distance and spawn data for both players.
    
    Additionally initializes the starting unit positions: each player starts at their
    base and their nearest neighbor to the center node.
    """
    nd, nset, dist, spawn = _compute_map_structure(treelines, nodes_list, bases)
    
    # Update global mutable state
    TREES.clear()
    TREES.update(treelines)
    
    NODES[:] = nodes_list
    
    # Rebuild the index mapping from cell -> node_index (excluding centre if handled separately)
    # Note: In original code, NODE_IDX mapped cell to index 0..N-1. 
    # We preserve this behavior but ensure it maps correctly for non-center nodes.
    NODE_IDX.clear()
    for i, c in enumerate(nodes_list):
        NODE_IDX[c] = i
    
    BASES[:] = bases
    
    # Update neighbor, distance, and spawn lists
    NEIGH_DIR[:] = nd
    NEIGH_SET[:] = nset
    DIST[:] = dist
    SPAWN[:] = spawn
    
    # Initialize starting units (base cell + best neighbor to center)
    p0_base = bases[0]
    center_node = nodes_list[0]
    
    # Find the valid neighbor of p0_base that is NOT a tree and is closest to center
    # We look in nd[p0_base], filtering for non-tree cells (>= 0) and excluding the base itself
    valid_p0_neighbors = [c for c in NEIGH_DIR[p0_base] if c >= 0 and c != p0_base]
    
    if valid_p0_neighbors:
        # Pick the neighbor with minimum distance to the centre node
        best_neighbor_p0 = min(valid_p0_neighbors, key=lambda c_idx: DIST[c_idx][center_node])
    else:
        # Fallback or error handling depending on game rules (usually shouldn't happen with valid maps)
        best_neighbor_p0 = p0_base
    
    START_UNITS[:] = [
        [p0_base, best_neighbor_p0],
        [bases[1], rot(best_neighbor_p0)]  # Player 1 gets rotated counterpart
    ]


def generate_map(seed):
    """Generates a random symmetric map.
    
    The map is guaranteed to be rotationally symmetric (180 degrees).
    It attempts up to 1000 times to find a configuration that satisfies:
      - Both players have at least 2 valid moves from their bases.
      - Bases are far enough apart (dist < 99, dist >= 5).
      - Every node is reachable from every base (connected graph).
      - The nearest non-center node to a base is not too far away (> 4 turns).
    
    Args:
        seed: Random seed for reproducibility.
        
    Returns:
        A tuple of (treelines, nodes_list, bases).
        
    Raises:
        RuntimeError: If no valid map can be generated after 1000 attempts.
    """
    rnd = random.Random(seed)
    # Available cells for placing trees, nodes, and bases (excluding centre)
    available_cells = [c for c in range(N) if c != CENTRE]
    
    for _ in range(1000):
        # 1. Pick base pair
        b0 = rnd.choice(available_cells)
        if rot(b0) == b0:
            continue
        b0_pair = (b0, rot(b0))
        
        # Remaining cells for trees and nodes
        remaining_cells = [c for c in available_cells if c not in b0_pair]
        
        # Select 4 pairs of symmetric cells from the remainder
        pick_pairs = rnd.sample([c for c in remaining_cells if c < rot(c)], k=2) # Need 4 cells total -> 2 pairs? 
        # Correction: We need 4 nodes (excluding centre). Let's pick 2 distinct symmetric pairs.
        valid_symmetric_choices = [c for c in remaining_cells if c < rot(c)]
        selected_pairs = rnd.sample(valid_symmetric_choices, 4) # Pick 4 individual cells that are already checked for symmetry
        
        trees = set()
        nodes_list = [CENTRE]
        
        # First pair -> Trees
        t1_pair = selected_pairs[0]
        t2_pair = selected_pairs[1]
        tree_cells = {t1_pair, rot(t1_pair), t2_pair, rot(t2_pair)}
        
        if b0 in tree_cells or rot(b0) in tree_cells:
            continue
            
        # Add node pairs (last two selections)
        node_cells_raw = selected_pairs[2]
        node_cells_raw_2 = selected_pairs[3]
        
        n1_pair = {node_cells_raw, rot(node_cells_raw)}
        n2_pair = {node_cells_raw_2, rot(node_cells_raw_2)}
        
        # Check for overlap between tree and node sets
        if any(n in tree_cells or n in b0_pair for n in (n1_pair | n2_pair)):
            continue
            
        nodes_list.extend(list(n1_pair))
        nodes_list.extend(list(n2_pair))
        
        # Set bases
        bases = [b0, rot(b0)]
        
        # Validate geometry using our structure computation helper
        nd_temp, _, dist_matrix, _ = _compute_map_structure(tree_cells, nodes_list, bases)
        
        # Check Player 0 reachability (needs >= 2 moves from base)
        if sum(1 for c in nd_temp[bases[0]] if c >= 0) < 2:
            continue
            
        # Check distance constraints
        dist_between_bases = dist_matrix[bases[0]][rot(bases[0])]
        if dist_between_bases >= INF or dist_between_bases < 5:
            continue
        
        # Check connectivity to all nodes from base
        if any(dist_matrix[bases[0]][n] >= INF for n in nodes_list):
            continue
            
        # Check that the closest non-center node isn't too close (>= 4 steps)
        # This was `min(dist[b0][n] for n in nodes[1:]) > 4` in original code
        min_dist_to_other_node = min((dist_matrix[bases[0]][n] for n in nodes_list[1:]), default=INF)
        if min_dist_to_other_node <= 4:
            continue
            
        return tree_cells, nodes_list, bases

    raise RuntimeError("could not generate a map")


def set_map(seed=None):
    """Initializes the game world.
    
    If seed is None, loads the default hand-crafted map.
    Otherwise, generates a random map with that seed.
    Finally, applies it to global state variables.
    """
    if seed is None:
        _apply_map(
            set(DEFAULT_TREES), 
            list(DEFAULT_NODES), 
            list(DEFAULT_BASES)
        )
    else:
        generated_trees, generated_nodes, generated_bases = generate_map(seed)
        _apply_map(generated_trees, generated_nodes, generated_bases)


# Initialize with default map immediately on import
set_map(None)


class State:
    """Mutable game state."""
    __slots__ = ("units", "res", "bhp", "post", "stock", "player", "ap", 
                 "turn", "over", "winner", "stoch", "hidden")

    def __init__(self, stoch=True, hidden=False):
        # Initialize unit arrays based on precomputed START_UNITS (base + spawn pos)
        self.units = [
            [[cell, UNIT_HP, 0] for cell in START_UNITS[0]],
            [[cell, UNIT_HP, 0] for cell in START_UNITS[1]]
        ]
        
        self.res = list(START_RES) * 2   # [player_0_resources, player_1_resources]
        self.bhp = list(BASE_HP) * 2     # [base_0_hp, base_1_hp]
        
        # Post: outpost ownership per node (-1 if contested/none)
        self.post = [-1] * len(NODES)
        
        # Stockpiles: Center node first, then others
        self.stock = [CENTER_STOCK] + [NODE_STOCK] * (len(NODES) - 1)
        
        self.player = 0      # Whose turn it is to take actions
        self.ap = AP         # Action points available
        
        self.turn = 0        # Count of completed turns (rounds passed)
        self.over = False
        self.winner = None   # 0, 1, or -1 for draw
        
        self.stoch = stoch   # Toggle random yields/combat
        self.hidden = hidden # Obscure enemy resources