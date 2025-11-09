import heapq

def state_check(state):
    """check the format of state, and return corresponding goal state.
       Do NOT edit this function."""
    non_zero_numbers = [n for n in state if n != 0]
    num_tiles = len(non_zero_numbers)
    if num_tiles == 0:
        raise ValueError('At least one number is not zero.')
    elif num_tiles > 9:
        raise ValueError('At most nine numbers in the state.')
    matched_seq = list(range(1, num_tiles + 1))
    if len(state) != 9 or not all(isinstance(n, int) for n in state):
        raise ValueError('State must be a list contain 9 integers.')
    elif not all(0 <= n <= 9 for n in state):
        raise ValueError('The number in state must be within [0,9].')
    elif len(set(non_zero_numbers)) != len(non_zero_numbers):
        raise ValueError('State can not have repeated numbers, except 0.')
    elif sorted(non_zero_numbers) != matched_seq:
        raise ValueError('For puzzles with X tiles, the non-zero numbers must be within [1,X], '
                          'and there will be 9-X grids labeled as 0.')
    goal_state = matched_seq
    for _ in range(9 - num_tiles):
        goal_state.append(0)
    return tuple(goal_state)

def get_manhattan_distance(from_state, to_state):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (The first one is current state, and the second one is goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    
    if len(from_state) != 9 or len(to_state) != 9:
        raise ValueError("States dont have 9 elements")
        
    distance = 0
    n = m = 3
    
    from_grid = [from_state[i:i+3] for i in range(0, 9, 3)]
    to_grid = [to_state[i:i+3] for i in range(0, 9, 3)]
    
    for i in range(n):
        for j in range(m):
            if from_grid[i][j] != 0:
                for i2 in range(n):
                    for j2 in range(m):
                        if to_grid[i2][j2] == from_grid[i][j]:
                            distance += abs(i2 - i) + abs(j2 - j)
                            
    return distance


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """

    # given state, check state format and get goal_state.
    goal_state = state_check(state)
    # please remove debugging prints when you submit your code.
    # print('initial state: ', state)
    # print('goal state: ', goal_state)

    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(list(succ_state)3, "h={}".format(get_manhattan_distance(succ_state,goal_state)))

def move(s, i, j, i2, j2, succ):
    if s[i][j] != 0 and s[i2][j2] == 0:
        next_state = [x[:] for x in s]
        next_state[i][j] = s[i2][j2]
        next_state[i2][j2] = s[i][j]
        succ.append(next_state)

def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    if len(state) != 9:
        raise ValueError("State must have 9 elements")
        
    n = m = 3
    s = [state[i:i+3] for i in range(0, 9, 3)]
    
    succ = []
    for i in range(n):
        for j in range(m):
            move(s, i, j, max(i - 1, 0), j, succ)
            move(s, i, j, min(i + 1, n - 1), j, succ)
            move(s, i, j, i, max(j - 1, 0), succ)
            move(s, i, j, i, min(j + 1, m - 1), succ)
    
    result = []
    for successor in succ:
        flat_successor = []
        for row in successor:
            flat_successor.extend(row)
        result.append(tuple(flat_successor))
    
    return sorted(result)



def solve(state):
    """
    A* algorithm implementation for the puzzle.
    """

    goal_state = state_check(state)
    pq = []
    closed = {}
    path_dict = {}
    state_info_list = []
    initial_state = tuple(state)
    goal_state = tuple(goal_state)
    initial_h = get_manhattan_distance(initial_state, goal_state)

    heapq.heappush(pq, (initial_h, initial_state, 0))
    closed[initial_state] = 0
    path_dict[initial_state] = None 
    max_length = 1
    solution_found = False

    while pq:
        curr_cost, curr_state, curr_g = heapq.heappop(pq)

        if curr_state == goal_state:
            solution_found = True
            break

        successors = get_succ(list(curr_state))

        for succ_state in successors:
            new_g = curr_g + 1
            if succ_state not in closed or new_g < closed[succ_state]:
                closed[succ_state] = new_g
                new_h = get_manhattan_distance(succ_state, goal_state)
                new_cost = new_g + new_h

                heapq.heappush(pq, (new_cost, succ_state, new_g))
                path_dict[succ_state] = curr_state

                state_info_list.append((list(succ_state), new_h, new_g))

        max_length = max(max_length, len(pq))

    if solution_found:
        print(True)
        path = []
        current = goal_state
        while current is not None:
            path.append(current)
            current = path_dict[current]
        path.reverse()

        for state in path:
            h = get_manhattan_distance(state, goal_state)
            moves = closed[state]
            print(list(state), f"h={h}", f"moves: {moves}")

        print(f"Max queue length: {max_length}")
    else:
        print(False)


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    # print()
    #
    # print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()

    #print(state_check([4,3,0,5,1,6,7,2,0])) STATE CHECK IS RIGHT AND WAS GIVEN
    #print(get_manhattan_distance([4,3,0,5,1,6,7,2,0], state_check([4,3,0,5,1,6,7,2,0]))) MANHATTAN DISTANCE IS RIGHT
    #print_succ([4,3,0,5,1,6,7,2,0]) PRINT SUCC WORKS CORRECTLY WHICH MEANS GET SUCC WORKS CORRECTLY
    #solve([4,3,0,5,1,6,7,2,0])
    #SOLVE FUNCTION LEFT
