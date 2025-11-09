import random

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def run_challenge_test(self):
        # Set to True if you would like to run gradescope against the challenge AI!
        # Leave as False if you would like to run the gradescope tests faster for debugging.
        # You can still get full credit with this set to False
        return False

    
    def heuristic_game_value(self, state):
        """
        Evaluates non-terminal states using heuristics.
        Returns a float between -1 and 1.
        """
        terminal_value = self.game_value(state)
        if terminal_value != 0:
            return terminal_value
        
        my_score = 0
        opp_score = 0
        
        directions = [
            [(0,1), (0,2), (0,3)],
            [(1,0), (2,0), (3,0)],
            [(1,1), (2,2), (3,3)],
            [(1,-1), (2,-2), (3,-3)],
            [(1,0), (0,1), (1,1)]
        ]
        
        for i in range(5):
            for j in range(5):
                if state[i][j] != ' ':
                    for direction in directions:
                        count_my = 0
                        count_opp = 0
                        valid_pattern = True
                        spaces = 0
                        checked_positions = [(i,j)]
                        
                        for di, dj in direction:
                            new_i, new_j = i + di, j + dj
                            if 0 <= new_i < 5 and 0 <= new_j < 5:
                                checked_positions.append((new_i, new_j))
                                if state[new_i][new_j] == self.my_piece:
                                    count_my += 1
                                elif state[new_i][new_j] == self.opp:
                                    count_opp += 1
                                elif state[new_i][new_j] == ' ':
                                    spaces += 1
                            else:
                                valid_pattern = False
                                break

                        if valid_pattern:
                            if state[i][j] == self.my_piece:
                                count_my += 1
                            else:
                                count_opp += 1
                                
                            if count_my > 0 and count_opp == 0 and spaces > 0:
                                my_score += (count_my / 4.0) ** 2
                            elif count_opp > 0 and count_my == 0 and spaces > 0:
                                opp_score += (count_opp / 4.0) ** 2
        
        center_weight = 0.1
        for i in range(5):
            for j in range(5):
                if state[i][j] == self.my_piece:
                    center_dist = abs(i - 2) + abs(j - 2)
                    my_score += center_weight * (4 - center_dist) / 4
                elif state[i][j] == self.opp:
                    center_dist = abs(i - 2) + abs(j - 2)
                    opp_score += center_weight * (4 - center_dist) / 4
        
        if my_score > 0 and opp_score > 0:
            return (my_score - opp_score) / max(my_score, opp_score)
        elif my_score > 0:
            return my_score / 2
        elif opp_score > 0:
            return -opp_score / 2
        return 0


    def max_value(self, state, depth, max_depth=3, alpha=float('-inf'), beta=float('inf')):
        if depth >= max_depth:
            return self.heuristic_game_value(state)
        
        terminal_value = self.game_value(state)
        if terminal_value != 0:
            return terminal_value
        
        value = float('-inf')
        for successor in self.succ(state):
            value = max(value, self.min_value(successor, depth + 1, max_depth, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def min_value(self, state, depth, max_depth=3, alpha=float('-inf'), beta=float('inf')):
        if depth >= max_depth:
            return self.heuristic_game_value(state)
        
        terminal_value = self.game_value(state)
        if terminal_value != 0:
            return terminal_value
        
        value = float('inf')
        for successor in self.succ(state):
            value = min(value, self.max_value(successor, depth + 1, max_depth, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        piece_count = sum(row.count(self.my_piece) for row in state)
        drop_phase = piece_count < 4

        successors = self.succ(state)
        if not successors:
            return None

        best_value = float('-inf')
        best_successor = None
        alpha = float('-inf')
        beta = float('inf')

        for successor in successors:
            value = self.min_value(successor, 0, max_depth=3, alpha=alpha, beta=beta)
            if value > best_value:
                best_value = value
                best_successor = successor
            alpha = max(alpha, best_value)

        if not best_successor:
            return None

        if drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] != best_successor[i][j]:
                        return [(i, j)]
        else:
            source_pos = None
            dest_pos = None
            for i in range(5):
                for j in range(5):
                    if state[i][j] != best_successor[i][j]:
                        if state[i][j] == self.my_piece:
                            source_pos = (i, j)
                        elif best_successor[i][j] == self.my_piece:
                            dest_pos = (i, j)
            
            if source_pos and dest_pos:
                return [dest_pos, source_pos]

        for i in range(5):
            for j in range(5):
                if state[i][j] == ' ':
                    return [(i, j)]

        return None

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # Check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i] == self.my_piece else -1

        # Check vertical wins
        for col in range(5):
            for i in range(2):
                if (state[i][col] != ' ' and 
                    state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]):
                    return 1 if state[i][col] == self.my_piece else -1

        # Check diagonal wins (top-left to bottom-right)
        for i in range(2):
            for j in range(2):
                if (state[i][j] != ' ' and
                    state[i][j] == state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3]):
                    return 1 if state[i][j] == self.my_piece else -1

        # Check diagonal wins (top-right to bottom-left)
        for i in range(2):
            for j in range(3, 5):
                if (state[i][j] != ' ' and
                    state[i][j] == state[i+1][j-1] == state[i+2][j-2] == state[i+3][j-3]):
                    return 1 if state[i][j] == self.my_piece else -1

        # Check 2x2 box wins
        for i in range(4):
            for j in range(4):
                if (state[i][j] != ' ' and
                    state[i][j] == state[i+1][j] == state[i][j+1] == state[i+1][j+1]):
                    return 1 if state[i][j] == self.my_piece else -1

        return 0

    

    def succ(self, state):
        successors = []
        
        def create_state_copy(state):
            return [[cell for cell in row] for row in state]

        piece_count = sum(row.count(self.my_piece) for row in state)
        drop_phase = piece_count < 4
        
        if drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == ' ':
                        successor = create_state_copy(state)
                        successor[i][j] = self.my_piece
                        successors.append(successor)
        else:
            adjacent_moves = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
            
            for i in range(5):
                for j in range(5):
                    if state[i][j] == self.my_piece:
                        for move in adjacent_moves:
                            new_i = i + move[0]
                            new_j = j + move[1]
                            
                            if (0 <= new_i < 5 and 0 <= new_j < 5 and 
                                state[new_i][new_j] == ' '):
                                successor = create_state_copy(state)
                                successor[i][j] = ' '
                                successor[new_i][new_j] = self.my_piece
                                successors.append(successor)

        return successors
    


   

    
    
    

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()