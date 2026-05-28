import numpy as np

# ---------------------------------------------------------------------------
# Numba check and decorator setup
# ---------------------------------------------------------------------------
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Mock decorator if numba is not available
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator
    # Mock prange
    prange = range

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA = np.array([0.0, 1.0, 2.0, 3.0])
BETA = np.array([0.0, 0.1, 0.3, 0.5])
GAMMA = 1.0

# ---------------------------------------------------------------------------
# Core Helper Functions (Numba-compatible)
# ---------------------------------------------------------------------------
@njit
def get_r_base(w):
    if w <= 0.2:
        return 4.0
    elif w <= 3.0:
        return 10.0
    elif w <= 10.0:
        return 15.0
    elif w <= 30.0:
        return 20.0
    else:
        return 30.0

@njit
def get_delivery_reward(w, et, p, t_delivery, T):
    rb = get_r_base(w)
    if t_delivery <= et:
        bonus = max(0.0, float(et - t_delivery) / max(float(et), 1.0))
        return ALPHA[p] * rb * (1.0 + bonus)
    else:
        factor = max(0.0, 1.0 - float(t_delivery - et) / max(float(T), 1.0))
        return BETA[p] * rb * factor

@njit
def get_move_cost(w_carried, W_max):
    return -0.01 * (1.0 + GAMMA * w_carried / max(W_max, 1.0))


# ---------------------------------------------------------------------------
# Trip Optimization Backtracking (Iterative or Numba recursion)
# ---------------------------------------------------------------------------
# DO NOT MODIFY: The block under `if HAS_NUMBA:` is optimized to the best performance for Numba.
if HAS_NUMBA:
    @njit
    def search_best_trip_sequence(
        start_node_idx,
        start_t,
        initial_w,
        initial_bag_size,
        trip_order_indices,
        dist_matrix,
        pickup_nodes,
        delivery_nodes,
        weights,
        deadlines,
        priorities,
        in_bags,
        W_max,
        K_max,
        T
    ):
        """
        Finds the optimal sequence of stops for a small subset of orders.
        Returns: (best_net_reward, end_node_idx, end_t, best_stops_array)
        Each stop in best_stops_array is encoded as: stop_node_idx + 100000 * (op) + 10000000 * (order_id)
        where:
          - op: 1 for pickup, 2 for delivery
        """
        n = len(trip_order_indices)
        if n == 0:
            return 0.0, start_node_idx, start_t, np.zeros(0, dtype=np.int64)

        # Pre-calculate active stops
        # We have up to 2n stops:
        #   stop 2r: pickup of trip_order_indices[r]
        #   stop 2r + 1: delivery of trip_order_indices[r]
        num_stops = 2 * n
        needed_stops = np.zeros(num_stops, dtype=np.int32)
        stop_nodes = np.zeros(num_stops, dtype=np.int32)
        stop_weights = np.zeros(num_stops, dtype=np.float64)
        stop_deadlines = np.zeros(num_stops, dtype=np.int32)
        stop_priorities = np.zeros(num_stops, dtype=np.int32)
        stop_ops = np.zeros(num_stops, dtype=np.int32)
        stop_order_ids = np.zeros(num_stops, dtype=np.int64)

        initial_visited_mask = 0
        total_needed_stops = 0

        for r in range(n):
            o_idx = trip_order_indices[r]
            # Pickup stop
            stop_nodes[2*r] = pickup_nodes[o_idx]
            stop_weights[2*r] = weights[o_idx]
            stop_deadlines[2*r] = deadlines[o_idx]
            stop_priorities[2*r] = priorities[o_idx]
            stop_ops[2*r] = 1
            stop_order_ids[2*r] = o_idx
            if in_bags[o_idx] == 1:
                initial_visited_mask |= (1 << (2*r))
            else:
                needed_stops[2*r] = 1
                total_needed_stops += 1

            # Delivery stop
            stop_nodes[2*r+1] = delivery_nodes[o_idx]
            stop_weights[2*r+1] = weights[o_idx]
            stop_deadlines[2*r+1] = deadlines[o_idx]
            stop_priorities[2*r+1] = priorities[o_idx]
            stop_ops[2*r+1] = 2
            stop_order_ids[2*r+1] = o_idx
            needed_stops[2*r+1] = 1
            total_needed_stops += 1

        # Search states stack (iterative DFS)
        # Stack size of 100 is plenty for depth <= 6
        stack_u = np.zeros(100, dtype=np.int32)
        stack_t = np.zeros(100, dtype=np.int32)
        stack_w = np.zeros(100, dtype=np.float64)
        stack_bag = np.zeros(100, dtype=np.int32)
        stack_mask = np.zeros(100, dtype=np.int32)
        stack_score = np.zeros(100, dtype=np.float64)
        stack_depth = np.zeros(100, dtype=np.int32)
        
        # To reconstruct the path, stack_path[depth] stores the stop index at that depth
        # Since we need to store the path for each stack element, we can use a flat history array
        stack_parent = np.zeros(100, dtype=np.int32)
        stack_last_stop = np.zeros(100, dtype=np.int32)

        # Initialize stack
        stack_u[0] = start_node_idx
        stack_t[0] = start_t
        stack_w[0] = initial_w
        stack_bag[0] = initial_bag_size
        stack_mask[0] = initial_visited_mask
        stack_score[0] = 0.0
        stack_depth[0] = 0
        stack_parent[0] = -1
        stack_last_stop[0] = -1

        stack_ptr = 1

        best_score = -1e9
        best_end_node = start_node_idx
        best_end_t = start_t
        
        # We will record the best path using parent pointers
        # To reconstruct the path of the best state, we store its index in a history log
        log_u = np.zeros(1000, dtype=np.int32)
        log_t = np.zeros(1000, dtype=np.int32)
        log_w = np.zeros(1000, dtype=np.float64)
        log_bag = np.zeros(1000, dtype=np.int32)
        log_mask = np.zeros(1000, dtype=np.int32)
        log_score = np.zeros(1000, dtype=np.float64)
        log_depth = np.zeros(1000, dtype=np.int32)
        log_parent = np.zeros(1000, dtype=np.int32)
        log_last_stop = np.zeros(1000, dtype=np.int32)
        log_ptr = 0

        while stack_ptr > 0:
            # Pop from stack
            stack_ptr -= 1
            curr_u = stack_u[stack_ptr]
            curr_t = stack_t[stack_ptr]
            curr_w = stack_w[stack_ptr]
            curr_bag = stack_bag[stack_ptr]
            curr_mask = stack_mask[stack_ptr]
            curr_score = stack_score[stack_ptr]
            curr_depth = stack_depth[stack_ptr]
            curr_parent = stack_parent[stack_ptr]
            curr_last_stop = stack_last_stop[stack_ptr]

            # Log current state to reconstruct path later
            if log_ptr < 1000:
                log_idx = log_ptr
                log_u[log_idx] = curr_u
                log_t[log_idx] = curr_t
                log_w[log_idx] = curr_w
                log_bag[log_idx] = curr_bag
                log_mask[log_idx] = curr_mask
                log_score[log_idx] = curr_score
                log_depth[log_idx] = curr_depth
                log_parent[log_idx] = curr_parent
                log_last_stop[log_idx] = curr_last_stop
                log_ptr += 1
            else:
                log_idx = -1

            # Check if all needed stops are visited
            all_visited = True
            for s_idx in range(num_stops):
                if needed_stops[s_idx] == 1 and (curr_mask & (1 << s_idx)) == 0:
                    all_visited = False
                    break

            if all_visited:
                if curr_score > best_score:
                    best_score = curr_score
                    best_end_node = curr_u
                    best_end_t = curr_t
                    best_log_idx = log_idx
                continue

            # Try to visit next stops
            for s_idx in range(num_stops):
                if needed_stops[s_idx] == 0 or (curr_mask & (1 << s_idx)) != 0:
                    continue

                # check validity
                is_pickup = (stop_ops[s_idx] == 1)
                o_id = stop_order_ids[s_idx]

                if is_pickup:
                    # Can pickup only if under capacity
                    if curr_bag >= K_max or curr_w + stop_weights[s_idx] > W_max:
                        continue
                    next_bag = curr_bag + 1
                    next_w = curr_w + stop_weights[s_idx]
                    reward_change = 0.0
                else:
                    # Can deliver only if pickup is already visited
                    pickup_s_idx = s_idx - 1
                    if (curr_mask & (1 << pickup_s_idx)) == 0:
                        continue
                    next_bag = curr_bag - 1
                    next_w = max(0.0, curr_w - stop_weights[s_idx])
                    
                # Travel cost
                nxt_node = stop_nodes[s_idx]
                dist = dist_matrix[curr_u, nxt_node]
                if dist >= 1e8:
                    continue
                
                next_t = curr_t + dist
                move_cost = dist * get_move_cost(curr_w, W_max)
                
                if not is_pickup:
                    # Delivery reward
                    reward_change = get_delivery_reward(
                        stop_weights[s_idx],
                        stop_deadlines[s_idx],
                        stop_priorities[s_idx],
                        next_t,
                        T
                    )
                
                next_score = curr_score + move_cost + reward_change

                # Push to stack
                if stack_ptr < 100:
                    stack_u[stack_ptr] = nxt_node
                    stack_t[stack_ptr] = next_t
                    stack_w[stack_ptr] = next_w
                    stack_bag[stack_ptr] = next_bag
                    stack_mask[stack_ptr] = curr_mask | (1 << s_idx)
                    stack_score[stack_ptr] = next_score
                    stack_depth[stack_ptr] = curr_depth + 1
                    stack_parent[stack_ptr] = log_idx
                    stack_last_stop[stack_ptr] = s_idx
                    stack_ptr += 1

        # Reconstruct path
        path_stops = []
        if best_score > -1e8:
            curr_log_idx = best_log_idx
            while curr_log_idx != -1:
                s_idx = log_last_stop[curr_log_idx]
                if s_idx != -1:
                    node = stop_nodes[s_idx]
                    op = stop_ops[s_idx]
                    oid = stop_order_ids[s_idx]
                    # Encode target: node + 100000 * op + 10000000 * oid
                    encoded = int(node) + 100000 * int(op) + 10000000 * int(oid)
                    path_stops.append(encoded)
                curr_log_idx = log_parent[curr_log_idx]
            path_stops.reverse()

        stops_arr = np.zeros(len(path_stops), dtype=np.int64)
        for idx, val in enumerate(path_stops):
            stops_arr[idx] = val

        return best_score, best_end_node, best_end_t, stops_arr


    # ---------------------------------------------------------------------------
    @njit
    def split_giant_tour_numba(
        start_node_idx,
        start_t,
        initial_w,
        initial_bag_size,
        giant_tour_indices,
        dist_matrix,
        pickup_nodes,
        delivery_nodes,
        weights,
        deadlines,
        priorities,
        in_bags,
        W_max,
        K_max,
        T,
        max_trip_size=3
    ):
        """
        Runs the Split DP algorithm to partition giant_tour_indices into optimal trips.
        Returns: (max_total_reward, path_stops_array)
        """
        m = len(giant_tour_indices)
        if m == 0:
            return 0.0, np.zeros(0, dtype=np.int64)

        # DP tables
        dp_score = np.full(m + 1, -1e9, dtype=np.float64)
        dp_pos = np.zeros(m + 1, dtype=np.int32)
        dp_time = np.zeros(m + 1, dtype=np.int32)
        dp_parent = np.full(m + 1, -1, dtype=np.int32)
        
        # Store the stops arrays for each DP transition
        dp_stops_offsets = np.zeros(m + 1, dtype=np.int32)
        dp_stops_lengths = np.zeros(m + 1, dtype=np.int32)
        stops_buffer = np.zeros(10000, dtype=np.int64)
        buffer_ptr = 0

        dp_score[0] = 0.0
        dp_pos[0] = start_node_idx
        dp_time[0] = start_t

        # Monotone Queue buffers (flat arrays)
        # Holds indices of DP states in the sliding window [j - max_trip_size, j - 1]
        q_indices = np.zeros(max_trip_size + 2, dtype=np.int32)
        q_upper_bounds = np.zeros(max_trip_size + 2, dtype=np.float64)

        for j in range(1, m + 1):
            dp_stops_offsets[j] = buffer_ptr
            
            # 1. Collect candidate predecessor states `i`
            num_cands = 0
            for k in range(1, max_trip_size + 1):
                i = j - k
                if i < 0:
                    break
                if dp_score[i] < -1e8:
                    continue
                
                # Calculate maximum possible reward for the segment [i:j]
                max_reward = 0.0
                for o_idx in range(i, j):
                    oid = giant_tour_indices[o_idx]
                    w_val = weights[oid]
                    if w_val <= 0.2:
                        rb = 4.0
                    elif w_val <= 3.0:
                        rb = 10.0
                    elif w_val <= 10.0:
                        rb = 15.0
                    elif w_val <= 30.0:
                        rb = 20.0
                    else:
                        rb = 30.0
                    max_reward += 3.0 * rb * 2.0
                
                q_indices[num_cands] = i
                q_upper_bounds[num_cands] = dp_score[i] + max_reward
                num_cands += 1

            if num_cands == 0:
                buffer_ptr += (max_trip_size * 2)
                continue

            # 2. Sort candidates in descending order of upper bounds (Monotone Queue)
            for idx in range(num_cands - 1):
                for kdx in range(idx + 1, num_cands):
                    if q_upper_bounds[idx] < q_upper_bounds[kdx]:
                        tmp_i = q_indices[idx]
                        q_indices[idx] = q_indices[kdx]
                        q_indices[kdx] = tmp_i
                        
                        tmp_ub = q_upper_bounds[idx]
                        q_upper_bounds[idx] = q_upper_bounds[kdx]
                        q_upper_bounds[kdx] = tmp_ub

            # 3. Process candidates using monotone queue structure
            for idx in range(num_cands):
                i = q_indices[idx]
                ub = q_upper_bounds[idx]

                # Pruning condition: if the best possible upper bound is worse than
                # or equal to current dp_score[j], prune remaining candidates.
                if ub <= dp_score[j]:
                    break

                curr_pos = dp_pos[i]
                curr_t = dp_time[i]
                trip_initial_w = initial_w if i == 0 else 0.0
                trip_initial_bag = initial_bag_size if i == 0 else 0
                trip_orders = giant_tour_indices[i:j]

                trip_score, trip_end_pos, trip_end_t, trip_stops = search_best_trip_sequence(
                    curr_pos,
                    curr_t,
                    trip_initial_w,
                    trip_initial_bag,
                    trip_orders,
                    dist_matrix,
                    pickup_nodes,
                    delivery_nodes,
                    weights,
                    deadlines,
                    priorities,
                    in_bags,
                    W_max,
                    K_max,
                    T
                )

                if trip_score > -1e8:
                    cand_score = dp_score[i] + trip_score
                    if cand_score > dp_score[j]:
                        dp_score[j] = cand_score
                        dp_pos[j] = trip_end_pos
                        dp_time[j] = trip_end_t
                        dp_parent[j] = i
                        
                        dp_stops_lengths[j] = len(trip_stops)
                        for k in range(len(trip_stops)):
                            if buffer_ptr + k < 10000:
                                stops_buffer[buffer_ptr + k] = trip_stops[k]

            buffer_ptr += (max_trip_size * 2)

        # Reconstruct the giant path
        all_stops = []
        curr = m
        while curr > 0:
            parent = dp_parent[curr]
            if parent == -1:
                break
            offset = dp_stops_offsets[curr]
            length = dp_stops_lengths[curr]
            
            seg_stops = []
            for k in range(length):
                seg_stops.append(stops_buffer[offset + k])
            seg_stops.reverse()
            for val in seg_stops:
                all_stops.append(val)
            curr = parent

        all_stops.reverse()
        
        stops_arr = np.zeros(len(all_stops), dtype=np.int64)
        for idx, val in enumerate(all_stops):
            stops_arr[idx] = val

        return dp_score[m], stops_arr

else:
    # ---------------------------------------------------------------------------
    # Optimized Pure-Python Fallback (Recursion instead of Numpy Stack)
    # ---------------------------------------------------------------------------
    def search_best_trip_sequence(
        start_node_idx,
        start_t,
        initial_w,
        initial_bag_size,
        trip_order_indices,
        dist_matrix,
        pickup_nodes,
        delivery_nodes,
        weights,
        deadlines,
        priorities,
        in_bags,
        W_max,
        K_max,
        T
    ):
        n = len(trip_order_indices)
        if n == 0:
            return 0.0, start_node_idx, start_t, np.zeros(0, dtype=np.int64)

        num_stops = 2 * n
        needed_stops = [0] * num_stops
        stop_nodes = [0] * num_stops
        stop_weights = [0.0] * num_stops
        stop_deadlines = [0] * num_stops
        stop_priorities = [0] * num_stops
        stop_ops = [0] * num_stops
        stop_order_ids = [0] * num_stops

        initial_visited_mask = 0
        total_needed_stops = 0

        for r in range(n):
            o_idx = trip_order_indices[r]
            # Pickup stop
            stop_nodes[2*r] = int(pickup_nodes[o_idx])
            stop_weights[2*r] = float(weights[o_idx])
            stop_deadlines[2*r] = int(deadlines[o_idx])
            stop_priorities[2*r] = int(priorities[o_idx])
            stop_ops[2*r] = 1
            stop_order_ids[2*r] = int(o_idx)
            if in_bags[o_idx] == 1:
                initial_visited_mask |= (1 << (2*r))
            else:
                needed_stops[2*r] = 1
                total_needed_stops += 1

            # Delivery stop
            stop_nodes[2*r+1] = int(delivery_nodes[o_idx])
            stop_weights[2*r+1] = float(weights[o_idx])
            stop_deadlines[2*r+1] = int(deadlines[o_idx])
            stop_priorities[2*r+1] = int(priorities[o_idx])
            stop_ops[2*r+1] = 2
            stop_order_ids[2*r+1] = int(o_idx)
            needed_stops[2*r+1] = 1
            total_needed_stops += 1

        best_score = -1e9
        best_end_node = start_node_idx
        best_end_t = start_t
        best_stops = []

        needed_indices = [idx for idx in range(num_stops) if needed_stops[idx] == 1]

        def dfs(curr_u, curr_t, curr_w, curr_bag, curr_mask, curr_score, path):
            nonlocal best_score, best_end_node, best_end_t, best_stops
            
            all_visited = True
            for idx in needed_indices:
                if not (curr_mask & (1 << idx)):
                    all_visited = False
                    break
            
            if all_visited:
                if curr_score > best_score:
                    best_score = curr_score
                    best_end_node = curr_u
                    best_end_t = curr_t
                    best_stops = list(path)
                return

            for s_idx in range(num_stops):
                if needed_stops[s_idx] == 0 or (curr_mask & (1 << s_idx)):
                    continue

                is_pickup = (stop_ops[s_idx] == 1)
                if is_pickup:
                    if curr_bag >= K_max or curr_w + stop_weights[s_idx] > W_max:
                        continue
                    next_bag = curr_bag + 1
                    next_w = curr_w + stop_weights[s_idx]
                else:
                    pickup_s_idx = s_idx - 1
                    if not (curr_mask & (1 << pickup_s_idx)):
                        continue
                    next_bag = curr_bag - 1
                    next_w = max(0.0, curr_w - stop_weights[s_idx])

                nxt_node = stop_nodes[s_idx]
                dist = dist_matrix[curr_u, nxt_node]
                if dist >= 1e8:
                    continue

                next_t = curr_t + dist
                move_cost = dist * (-0.01 * (1.0 + curr_w / max(W_max, 1.0)))

                reward_change = 0.0
                if not is_pickup:
                    w_val = stop_weights[s_idx]
                    et_val = stop_deadlines[s_idx]
                    p_val = stop_priorities[s_idx]
                    
                    if w_val <= 0.2:
                        rb = 4.0
                    elif w_val <= 3.0:
                        rb = 10.0
                    elif w_val <= 10.0:
                        rb = 15.0
                    elif w_val <= 30.0:
                        rb = 20.0
                    else:
                        rb = 30.0

                    if next_t <= et_val:
                        bonus = max(0.0, float(et_val - next_t) / max(float(et_val), 1.0))
                        reward_change = ALPHA[p_val] * rb * (1.0 + bonus)
                    else:
                        factor = max(0.0, 1.0 - float(next_t - et_val) / max(float(T), 1.0))
                        reward_change = BETA[p_val] * rb * factor

                encoded = int(nxt_node) + 100000 * int(stop_ops[s_idx]) + 10000000 * int(stop_order_ids[s_idx])
                dfs(nxt_node, next_t, next_w, next_bag, curr_mask | (1 << s_idx), curr_score + move_cost + reward_change, path + [encoded])

        dfs(start_node_idx, start_t, initial_w, initial_bag_size, initial_visited_mask, 0.0, [])
        
        stops_arr = np.zeros(len(best_stops), dtype=np.int64)
        for idx, val in enumerate(best_stops):
            stops_arr[idx] = val
        return best_score, best_end_node, best_end_t, stops_arr

    def split_giant_tour_numba(
        start_node_idx,
        start_t,
        initial_w,
        initial_bag_size,
        giant_tour_indices,
        dist_matrix,
        pickup_nodes,
        delivery_nodes,
        weights,
        deadlines,
        priorities,
        in_bags,
        W_max,
        K_max,
        T,
        max_trip_size=3
    ):
        m = len(giant_tour_indices)
        if m == 0:
            return 0.0, np.zeros(0, dtype=np.int64)

        dp_score = [-1e9] * (m + 1)
        dp_pos = [0] * (m + 1)
        dp_time = [0] * (m + 1)
        dp_parent = [-1] * (m + 1)
        dp_stops = [None] * (m + 1)

        dp_score[0] = 0.0
        dp_pos[0] = int(start_node_idx)
        dp_time[0] = int(start_t)

        for j in range(1, m + 1):
            # 1. Collect candidates
            candidates = []
            upper_bounds = []
            for k in range(1, max_trip_size + 1):
                i = j - k
                if i < 0:
                    break
                if dp_score[i] < -1e8:
                    continue
                
                # Calculate max possible reward for the segment [i:j]
                max_reward = 0.0
                for o_idx in range(i, j):
                    oid = giant_tour_indices[o_idx]
                    w_val = weights[oid]
                    if w_val <= 0.2:
                        rb = 4.0
                    elif w_val <= 3.0:
                        rb = 10.0
                    elif w_val <= 10.0:
                        rb = 15.0
                    elif w_val <= 30.0:
                        rb = 20.0
                    else:
                        rb = 30.0
                    max_reward += 3.0 * rb * 2.0
                
                candidates.append(i)
                upper_bounds.append(dp_score[i] + max_reward)

            # 2. Sort candidates in descending order of upper bounds (Monotone Queue)
            # Zip and sort
            sorted_cands = sorted(zip(upper_bounds, candidates), key=lambda x: x[0], reverse=True)

            # 3. Process candidates
            for ub, i in sorted_cands:
                # Pruning condition
                if ub <= dp_score[j]:
                    break

                curr_pos = dp_pos[i]
                curr_t = dp_time[i]
                trip_initial_w = initial_w if i == 0 else 0.0
                trip_initial_bag = initial_bag_size if i == 0 else 0
                trip_orders = giant_tour_indices[i:j]

                trip_score, trip_end_pos, trip_end_t, trip_stops = search_best_trip_sequence(
                    curr_pos,
                    curr_t,
                    trip_initial_w,
                    trip_initial_bag,
                    trip_orders,
                    dist_matrix,
                    pickup_nodes,
                    delivery_nodes,
                    weights,
                    deadlines,
                    priorities,
                    in_bags,
                    W_max,
                    K_max,
                    T
                )

                if trip_score > -1e8:
                    cand_score = dp_score[i] + trip_score
                    if cand_score > dp_score[j]:
                        dp_score[j] = cand_score
                        dp_pos[j] = trip_end_pos
                        dp_time[j] = trip_end_t
                        dp_parent[j] = i
                        dp_stops[j] = trip_stops

        all_stops = []
        curr = m
        while curr > 0:
            parent = dp_parent[curr]
            if parent == -1:
                break
            trip_stops = dp_stops[curr]
            if trip_stops is not None:
                for val in reversed(trip_stops):
                    all_stops.append(val)
            curr = parent

        all_stops.reverse()
        
        stops_arr = np.zeros(len(all_stops), dtype=np.int64)
        for idx, val in enumerate(all_stops):
            stops_arr[idx] = val

        return dp_score[m], stops_arr


@njit(parallel=True)
def split_multiple_giant_tours_numba(
    start_node_idx,
    start_t,
    initial_w,
    initial_bag_size,
    giant_tours,
    dist_matrix,
    pickup_nodes,
    delivery_nodes,
    weights,
    deadlines,
    priorities,
    in_bags,
    W_max,
    K_max,
    T,
    max_trip_size
):
    num_tours = giant_tours.shape[0]
    scores = np.zeros(num_tours, dtype=np.float64)
    for t_idx in prange(num_tours):
        tour = giant_tours[t_idx]
        score, _ = split_giant_tour_numba(
            start_node_idx,
            start_t,
            initial_w,
            initial_bag_size,
            tour,
            dist_matrix,
            pickup_nodes,
            delivery_nodes,
            weights,
            deadlines,
            priorities,
            in_bags,
            W_max,
            K_max,
            T,
            max_trip_size
        )
        scores[t_idx] = score
    return scores


