import random

class GamePlayer:

    def __init__(self):
        pass
        # Result of -1 is black, result of 1 is white

    def playGame(self):
        boards_seen = []
        moves_made = 0
        is_black = False
        current_board = [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]]

        # Play out Game
        while moves_made <= 9: # 9 possible squares to fill
            # Check if has won
            win_state = self.hasWon(current_board)
            if win_state == 0:
                # get available squares to move to
                available_move_squares = self.getValidMoves(current_board)
                # get random square of available to move to
                random_move_square = random.randint(len(available_move_squares))
                # make the move and set the current board to the returned one
                current_board = self.makeMove(current_board, random_move_square, is_black)
                # add moves made and add board to boards seen
                moves_made +=1; boards_seen.append(current_board)
                # swap current player
                is_black = not is_black
            elif win_state == 1:
                return [boards_seen, [1]]
            elif win_state == -1:
                return [boards_seen, [-1]]

        return [boards_seen, [result]]

    def makeMove(self, current_board, place, is_black): # matrix goes in rows from top left, across, then down to next row (0-8)
        current_board_2 = current_board.copy()
        counter = 0
        for row in current_board_2:
            for square in row:
                if counter == place:
                    if square == 0:
                        if is_black:
                            current_board_2[counter] = -1
                        else:
                            current_board_2[counter] = 1
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
                counter += 0
        return valid_squares

    def hasWon(self, current_board):

        print(current_board[0, 0])

        # Columns
        if current_board[0, 0] == current_board[1, 0] == current_board[2, 0]:
            return current_board[0, 0]
        elif current_board[0, 1] == current_board[1, 1] == current_board[2, 1]:
            return current_board[0, 1]
        elif current_board[0, 2] == current_board[1, 2] == current_board[2, 2]:
            return current_board[0, 2]
        # Rows
        elif current_board[0, 0] == current_board[0, 1] == current_board[0, 2]:
            return current_board[0, 0]
        elif current_board[1, 0] == current_board[1, 1] == current_board[1, 2]:
            return current_board[1, 0]
        elif current_board[2, 0] == current_board[2, 1] == current_board[2, 2]:
            return current_board[2, 0]

        # Diagonals
        elif current_board[0, 0] == current_board[1, 1] == current_board[2, 2]:
            return current_board[0, 0]
        elif current_board[2, 0] == current_board[1, 1] == current_board[0, 2]:
            return current_board[2, 0]

        return 0


class Generator:

    game_player = GamePlayer()
    completed_games = []  # this array ->
    # Games[Single Game([[Game Matrix], [Game Matrix]], [value])]

    def __init__(self):
        print("Created Generator")
        result1 = self.game_player.playGame()
        print(result1)


