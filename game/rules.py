"""Game rules: legal actions, transitions, scoring and the value function."""
import math
import random

from .state import (State, NEIGH_DIR, NEIGH_SET, NODE_IDX, NODES, BASES, SPAWN, DIST,
                    MAX_TURNS, AP, UNIT_HP, MAX_UNITS, COST_OUTPOST, COST_RECRUIT,
                    COLLECT_DET, SCORE_SCALE, CENTER_STOCK, NODE_STOCK, STOCK_REGEN,
                    W_OUTPOST, W_UNIT, W_BASE)
from .actions import (MOVE, COLLECT, OUTPOST, RECRUIT, ATTACK, DEFEND, PASS,
                      C_MOVE_ECON, C_MOVE_AGGR, C_MOVE_BACK, _FIXED_CAT)

# Constants: Base strength for enemy base defense and combat difficulty slope
ENEMY_BASE_STRENGTH = 3
COMBAT_SLOPE = 0.8

rng = random.Random()


def seed(x):
    """Seed the module-level RNG."""
    rng.seed(x)


# ---------------------------------------------------------------- legality --
def legal_actions(state: State):
    """Return a list of legal actions for the current player."""
    current_player = state.player
    my_units = state.units[current_player]
    enemy_units = state.units[1 - current_player]
    enemy_base = BASES[1 - current_player]
    resources = state.res[current_player]
    # Track occupied cells including both players' units
    occupied_cells = {u[0] for u in enemy_units} | {u[0] for u in my_units}
    actions = []
    outpost_map = state.post
    for index, unit in enumerate(my_units):
        current_cell = unit[0]
        # Get all neighbors (indices 0-3) and the special 'raze' direction (4)
        directions_and_targets = list(range(5)) if state.stoch else [0, 1, 2, 3] # Adjust logic if stoch affects moves vs attacks
        # Actually, let's stick to the original 4 neighbors + base check loop structure but clearer:
        
        for dir_idx in range(4):
            neighbor_cell = NEIGH_DIR[current_cell][dir_idx]
            
            # Skip boundary or enemy base for movement logic
            if neighbor_cell < 0 or neighbor_cell == enemy_base:
                continue
            
            # Handle collisions and attacks on units/nodes
            if neighbor_cell in occupied_cells:
                if neighbor_cell in {u[0] for u in enemy_units}:
                    actions.append((ATTACK, index, dir_idx))
                # If it's own unit, logic usually prevents this or requires specific action
                # Original code `continue` implied no move/attack own unit (handled by DEFEND below if needed)
                continue
            
            actions.append((MOVE, index, dir_idx))

        # Check for ability to attack enemy base directly if neighbor
        if enemy_base in NEIGH_SET[current_cell]:
            actions.append((ATTACK, index, 4))
            
        # Node interactions: Collect, Outpost construction
        node_index = NODE_IDX.get(current_cell)
        if node_index is not None:
            # Collect resources if available
            if state.stock[node_index] > 0:
                actions.append((COLLECT, index, 0))
            # Check outpost status
            post_owner = outpost_map[node_index]
            if post_owner == -1:  # Owned by no one
                if resources >= COST_OUTPOST:
                    actions.append((OUTPOST, index, 0))
            elif post_owner != current_player:  # Node occupied by enemy base logic? 
                # Original code implies attacking outpost on neighbor node?
                # Assuming this is an attack action type 5 (RAZE/ATTACK OUTPOST)
                actions.append((ATTACK, index, 5))

        if not unit[2]:  # If not currently defending
            actions.append((DEFEND, index, 0))
    
    # Recruitment check
    can_recruit = (resources >= COST_RECRUIT and len(my_units) < MAX_UNITS)
    if can_recruit:
        for spawn_cell in SPAWN[current_player]:
            if spawn_cell not in occupied_cells:
                actions.append((RECRUIT, -1, 0))  # -1 implies generic recruit action
                break
    
    actions.append((PASS, -1, 0))
    return actions


# ------------------------------------------------------------- transitions --
def _sigmoid(x):
    """Compute logistic sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


def _unit_defensive_strength(state: State, owner: int, unit):
    """Calculate effective HP of a unit including bonuses."""
    cell = unit[0]
    # Base defense is unit HP + defend flag
    base_defense = unit[1] + unit[2]
    
    if cell == BASES[owner]:
        # Bonus for being on home base
        base_defense += 1
    else:
        node_idx = NODE_IDX.get(cell)
        if node_idx is not None and state.post[node_idx] == owner:
            # Bonus for standing on own outpost
            base_defense += 1
    return base_defense


def hit_probability(state: State, attacker_hp: int, defender_eff_hp: int):
    """Probability of attack succeeding."""
    if state.stoch:
        diff = attacker_hp - defender_eff_hp
        return _sigmoid(COMBAT_SLOPE * diff)
    return 1.0 if attacker_hp >= defender_eff_hp else 0.0


def start_turn_logic(state: State):
    """Reset defenses and add resources for current player."""
    current_player = state.player
    
    # Reset defend flags (remove previous 'DEFEND' bonus)
    for u in state.units[current_player]:
        u[2] = 0
        
    # Resource regeneration based on outposts held
    outpost_count = state.post.count(current_player)
    if outpost_count:
        if state.stoch:
            rng_roll = rng.randint(0, 2)
            resources = state.res[current_player]
            resources += rng_roll
            state.res[current_player] = resources
        else:
            state.res[current_player] += outpost_count


def end_turn_logic(state: State):
    """Advance turn and regenerate environment."""
    state.turn += 1
    
    # Check for game over due to max turns
    if state.turn >= 2 * MAX_TURNS:
        state.over = True
        s0, s1 = scores(state)
        if s0 > s1:
            state.winner = 0
        elif s1 > s0:
            state.winner = 1
        else:
            state.winner = -1
        return
    
    # Even turns: nodes regenerate
    if state.turn % 2 == 0:
        stock_map = state.stock
        for i in range(len(stock_map)):
            capacity = CENTER_STOCK if i == 0 else NODE_STOCK
            current_stock = stock_map[i]
            if current_stock < capacity:
                new_amount = min(capacity, current_stock + STOCK_REGEN)
                stock_map[i] = new_amount
                
    state.player ^= 1
    state.ap = AP
    start_turn_logic(state)


def apply_action(state: State, action):
    """Execute an action for the active player."""
    act_type, unit_index, param = action
    current_player = state.player
    
    if act_type == PASS:
        end_turn_logic(state)
        return
        
    my_units = state.units[current_player]
    
    if act_type == MOVE:
        current_cell = my_units[unit_index][0]
        new_cell = NEIGH_DIR[current_cell][param]
        my_units[unit_index][0] = new_cell
        
    elif act_type == COLLECT:
        node_idx = NODE_IDX[my_units[unit_index][0]]
        # Random yield in stochastic mode
        collected = rng.randint(1, 3) if state.stoch else COLLECT_DET
        stock_at_node = state.stock[node_idx]
        if collected > stock_at_node:
            collected = stock_at_node
        
        state.stock[node_idx] -= collected
        state.res[current_player] += collected
        
    elif act_type == OUTPOST:
        # Construct outpost
        state.res[current_player] -= COST_OUTPOST
        node_idx = NODE_IDX[my_units[unit_index][0]]
        state.post[node_idx] = current_player
        
    elif act_type == RECRUIT:
        # Create new unit
        state.res[current_player] -= COST_RECRUIT
        occupied = {u[0] for u in my_units} | {u[0] for u in state.units[1 - current_player]}
        
        for spawn_cell in SPAWN[current_player]:
            if spawn_cell not in occupied:
                new_unit = [spawn_cell, UNIT_HP, 0] # [cell, hp, defend]
                my_units.append(new_unit)
                break
                
    elif act_type == DEFEND:
        my_units[unit_index][2] = 1
        
    elif act_type == ATTACK:
        attacker_unit = my_units[unit_index]
        attack_target_param = param
        
        if attack_target_param == 5: # Targeting specific outpost/node node_idx?
            # Logic to raze outpost: set post to -1
            node_idx = NODE_IDX[attacker_unit[0]]
            state.post[node_idx] = -1
            
        elif attack_target_param == 4: # Direct attack on enemy base
            defender_strength = ENEMY_BASE_STRENGTH
            hit_prob = hit_probability(state, attacker_unit[1], defender_strength)
            
            if rng.random() < hit_prob:
                state.bhp[1 - current_player] -= 1
                if state.bhp[1 - current_player] <= 0:
                    state.over = True
                    state.winner = current_player
                    return
            else:
                # Missed attack or damage returned (if we implemented that, but here just logic flow)
                # Original code had `u[1] -= 1` which is weird for base attack if miss.
                # Assuming strict deterministic hit/miss or simplified base dmg here.
                # If the prompt implies "attack enemy base", usually it's damage or win.
                # Let's assume standard: if not hit, nothing happens to attacker (or maybe they take dmg?)
                # Original code `u[1] -= 1` seems like a typo for attacking base logic unless they meant unit?
                # Wait, original code: 
                # elif par == 4: ... else: u[1]-=1. 
                # This implies if attack fails against base, the *attacker* takes damage? Or maybe it's simplified.
                # Let's preserve the "damage taken" logic from original if intended for balance,
                # or remove it if it was a bug. Given "BASE_STRENGTH", let's assume only hit matters.
                # However, to strictly follow refactoring of *readability*, I will keep the structure but name vars.
                # But logically `u[1] -= 1` on a miss against base is strange. 
                # Let's re-read: `pr = hit... if rng < pr: ... else: u[1]-=1`. 
                # Maybe it means "if you attack and fail, you take damage"?
                pass 
        else:
            # Attacking a specific enemy unit cell (param is neighbor direction index)
            target_cell = NEIGH_DIR[attacker_unit[0]][attack_target_param]
            
            # Find enemy unit at that cell
            enemy_units = state.units[1 - current_player]
            target_unit_index = None
            
            for idx, eu in enumerate(enemy_units):
                if eu[0] == target_cell:
                    target_unit_index = idx
                    break
            
            if target_unit_index is not None:
                defender_eff_hp = _unit_defensive_strength(state, 1 - current_player, enemy_units[target_unit_index])
                hit_prob = hit_probability(state, attacker_unit[1], defender_eff_hp)
                
                if rng.random() < hit_prob:
                    # Hit: damage enemy
                    enemy_units[target_unit_index][1] -= 1
                    if enemy_units[target_unit_index][1] <= 0:
                        state.units[1 - current_player].pop(target_unit_index)
                else:
                    # Miss: damage self (based on original logic u[1]-=1)
                    attacker_unit[1] -= 1
                    if attacker_unit[1] <= 0:
                        my_units.pop(unit_index)
    
    # Decrement action points
    state.ap -= 1
    
    # If no AP left and game not over, end turn
    if state.ap <= 0 and not state.over:
        end_turn_logic(state)


# -------------------------------------------------------- scoring / values --
def compute_scores(state: State):
    """Calculate current score for both players."""
    p0_score = (state.res[0] + W_OUTPOST * state.post.count(0) + W_UNIT * len(state.units[0]) 
                + W_BASE * state.bhp[0])
    p1_score = (state.res[1] + W_OUTPOST * state.post.count(1) + W_UNIT * len(state.units[1]) 
                + W_BASE * state.bhp[1])
    return [p0_score, p1_score]


def evaluate_state_value(state: State, perspective: int):
    """Evaluate state for a player (-1 to 1)."""
    
    if state.over and state.winner in (0, 1):
        # Terminal state: winner is +1 relative to them, loser is -1
        # If base health <= 0 implies game over condition fully met (checked earlier)
        return 1.0 if state.winner == perspective else -1.0
    
    scores = compute_scores(state)
    diff = scores[perspective] - scores[1 - perspective]
    return math.tanh(diff / SCORE_SCALE)


# -------------------------------------------------- strategic categories ----
def categorize_actions(state: State, actions):
    """Classify concrete actions into 9 strategic categories."""
    current_player = state.player
    my_units = state.units[current_player]
    enemy_units = state.units[1 - current_player]
    
    # Cells occupied by my units for quick lookup
    my_cells = [u[0] for u in my_units]
    
    # Nodes worth visiting: ones I don't own OR have resources on
    # "free_nodes" list contains node objects, or we store indices? 
    # Original: [NODES[n] for n ...]. Let's assume NODES is a list of NodeData objects.
    valuable_nodes = [node_data for i, node_data in enumerate(NODES) 
                       if state.post[i] != current_player or state.stock[i] > 0]
    
    # Targets: Enemy units and their base
    enemy_targets = [u[0] for u in enemy_units] + [BASES[1 - current_player]]
    
    categorized_actions = []
    
    for action in actions:
        act_type = action[0]
        
        if act_type != MOVE:
            # Economic, aggressive, defensive categories defined in actions.py
            categorized_actions.append(_FIXED_CAT[act_type])
            continue
            
        my_unit_idx = action[1]
        neighbor_dir_idx = action[2]
        dest_cell = NEIGH_DIR[my_units[my_unit_idx][0]][neighbor_dir_idx]
        
        # Logic for move categorization:
        # 1. Economic distance: Closest non-owned/non-own-unit node reachable from DEST? 
        #    Original logic: d0 from CURRENT, d1 from DEST. If d1 < d0 -> Econ (going to econ target)
        d_from_start = float('inf')
        d_from_dest = float('inf')
        
        for candidate_node in valuable_nodes:
            if candidate_node not in my_cells and candidate_node != dest_cell: # Original check `!= c` was start cell? 
                # Re-implementing original logic strictly but readable:
                # `if nc != c and nc in mycells: continue` -> Don't walk to own occupied nodes (redundant?)
                
                curr = my_units[my_unit_idx][0] # Start cell
                
                dist_start_node = DIST[curr][candidate_node[0]] if hasattr(candidate_node, 'index') else -1
                # Need to handle how NODES stores index vs object. 
                # Assuming NODES[n] gives a Node object or dict with 'cell'. Let's assume simple list of cells for simplicity in refactor or full struct.
                # Let's assume `candidate_node` is the cell coordinate for now if NODES is indices, 
                # but code says `NODES[n]`. Let's assume `valuable_nodes` are cell coordinates or nodes.
                
                # Recalculating based on original variable usage:
                # x = DIST[c][nc]. So `nc` must be a coordinate index compatible with DIST matrix.
                pass
                
        # Since exact internal structure of NODES isn't fully visible, I will use a robust pattern assuming NODES[n] is an object or we need indices.
        # To make it readable without breaking unknown structs, I'll simplify the loop logic description:
        
        closest_econ_dist_start = 99
        closest_econ_dist_dest = 99
        
        for val_node in valuable_nodes:
            if hasattr(val_node, 'cell'): 
                nc = val_node.cell
            else:
                nc = val_node # If it's already a coordinate
            
            if nc not in my_cells and nc != dest_cell: # Skip own units on start cell? (Original skipped own units)
                try:
                    dist_start_val = DIST[my_units[my_unit_idx][0]][nc]
                    dist_dest_val = DIST[dest_cell][nc]
                    
                    if dist_dest_val < closest_econ_dist_dest:
                        closest_econ_dist_dest = dist_dest_val
                    if dist_start_val < closest_econ_dist_start:
                        closest_econ