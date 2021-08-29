import random

class GamePlayer:

    def __init__(self):
        pass
        # Result of -1 is black, result of 1 is white

    def playGame(self):
        boards_seen = []
        is_black = False
        current_board = [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]]

        # Play out Game
        while len(self.getValidMoves(current_board)) != 0:  # 9 possible squares to fill
            # Check if has won
            win_state = self.hasWon(current_board)
            if win_state == 0:
                # get available squares to move to
                available_move_squares = self.getValidMoves(current_board)
                # get random square of available to move to
                random_index_of_available_move_squares = random.randint(0, len(available_move_squares)-1) # BOTH BOUNDS INCLUDED...
                # make the move and set the current board to the returned one
                current_board = self.makeMove(current_board, available_move_squares[random_index_of_available_move_squares], is_black)
                # add moves made and add board to boards seen
                boards_seen.append(self.getBoardCopy(current_board))
                # swap current player
                is_black = not is_black
            elif win_state == 1:
                return [boards_seen, [1]]
            elif win_state == -1:
                return [boards_seen, [-1]]

        win_state = self.hasWon(current_board)
        if win_state == 1:
            return [boards_seen, [1]]
        elif win_state == -1:
            return [boards_seen, [-1]]
        else:
            return [boards_seen, [0]]

    def getBoardCopy(self, current_board):
        copy_board = [[], [], []]
        for row_idx in range(len(current_board)):
            for item_idx in range(len(current_board[row_idx])):
                copy_board[row_idx].append(current_board[row_idx][item_idx])
        return copy_board

    def makeMove(self, current_board, place, is_black): # matrix goes in rows from top left, across, then down to next row (0-8)
        current_board_2 = current_board.copy()
        counter = 0
        for row_idx in range(len(current_board_2)):
            for square_idx in range(len(current_board_2[row_idx])):
                if counter == place:
                    if current_board_2[row_idx][square_idx] == 0:
                        if is_black:
                            current_board_2[row_idx][square_idx] = -1
                        else:
                            current_board_2[row_idx][square_idx] = 1
                        return current_board_2
                    else:
                        return None # square already taken
                counter += 1
        return None

    def getValidMoves(self, current_board):
        valid_squares = []
        counter = 0
        for row in current_board:
            for square in row:
                if square == 0:
                    valid_squares.append(counter)
                counter += 1
        return valid_squares

    def hasWon(self, current_board):
        # Columns
        if current_board[0][0] == current_board[1][0] == current_board[2][0] and current_board[0][0] != 0:
            return current_board[0][0]
        elif current_board[0][1] == current_board[1][1] == current_board[2][1] and current_board[0][1] != 0:
            return current_board[0][1]
        elif current_board[0][2] == current_board[1][2] == current_board[2][2] and current_board[0][2] != 0:
            return current_board[0][2]
        # Rows
        elif current_board[0][0] == current_board[0][1] == current_board[0][2] and current_board[0][0] != 0:
            return current_board[0][0]
        elif current_board[1][0] == current_board[1][1] == current_board[1][2] and current_board[1][0] != 0:
            return current_board[1][0]
        elif current_board[2][0] == current_board[2][1] == current_board[2][2] and current_board[2][0] != 0:
            return current_board[2][0]

        # Diagonals
        elif current_board[0][0] == current_board[1][1] == current_board[2][2] and current_board[0][0] != 0:
            return current_board[0][0]
        elif current_board[2][0] == current_board[1][1] == current_board[0][2] and current_board[2][0] != 0:
            return current_board[2][0]

        return 0

    def drawBoard(self, current_board):
        counter = 0
        for row in current_board:
            for space in row:
                if counter % 3 == 0 and counter != 0:
                    print()
                print(space, end=" | ")
                counter += 1


class Generator:

    game_player = GamePlayer()
    completed_games = []  # this array ->
    # Games[Single Game([[Game Matrix], [Game Matrix]], [value])]

    def __init__(self, num_games=1_000_000):
        print("Created Generator")
        self.num_games = num_games

        # Run Games
        for game_idx in range(self.num_games):
            self.completed_games.append(self.game_player.playGame())
            if game_idx % 10000 == 0:
                print(game_idx)

        # Print First Game Sequence For Verification
        result1 = self.completed_games[0]
        for result in result1[0]:
            self.game_player.drawBoard(result)
            print()
            print("================================")
        print(f"Result = {result1[1][0]}")
