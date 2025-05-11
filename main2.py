import copy  # imports
import random
import time
import queue
import numpy as np
import pygame as pg
import itertools
from typing import Tuple
from collections import deque
import threading
import functools

#constants
colors = {
    'white': (213, 213, 213),
    'black': (25, 25, 25),
    'blue': (5, 5, 65),
    'l-blue': (91, 220, 255),
    'green1': (15, 100, 15),
    'green2': (45, 150, 45),
    'green3': (65, 200, 65),
    'gray': (100, 100, 100),
    'yellow': (255, 213, 0)
}

BOARD_SIZE = 8
SQUARE_SIZE = 75
HALF_SQUARE_SIZE = SQUARE_SIZE // 2
FONT_SIZE = 25


EMPTY = 0
BLACK = 1
WHITE = -1
pg.init()
s_font = pg.font.Font(None, FONT_SIZE)

class OthelloBoard:
    DIRECTIONS = list(itertools.product((-1, 0, 1), (-1, 0, 1)))

    def __init__(self):
        self.board = np.zeros((8, 8))
        self.board[3, 3], self.board[4, 4] = 1, 1
        self.board[3, 4], self.board[4, 3] = -1, -1
        self.empty_spaces = {(i, j) for i in range(8) for j in range(8) if self.board[i, j] == 0}
        self.current_player = 1
        self.turn_count = 0
        self.moves_skipped = 0
        self.moves_stack = deque()
        self.board_score = 0
        self.valid_moves = self.get_valid_moves()
        self.best_move = self.get_best_move(1)

    def __copy__(self):
        new_board = OthelloBoard()
        new_board.board = copy.deepcopy(self.board)
        new_board.empty_spaces = copy.deepcopy(self.empty_spaces)
        new_board.current_player = self.current_player
        new_board.turn_count = self.turn_count
        new_board.moves_skipped = self.moves_skipped
        move_stack_copy = self.moves_stack
        new_board.moves_stack = copy.deepcopy(move_stack_copy)
        new_board.valid_moves = copy.deepcopy(self.valid_moves)
        new_board.best_move = self.best_move
        new_board.board_score = self.board_score
        return new_board

    def __hash__(self):
        board_hash = hash(self.board.tobytes())
        empty_spaces_hash = hash(frozenset(self.empty_spaces))
        moves_stack_hash = hash(tuple((x, y, tuple(z)) for x, y, z in self.moves_stack if z is not None))
        return hash((board_hash, empty_spaces_hash,
                     self.current_player,
                     self.turn_count,
                     self.moves_skipped,
                     moves_stack_hash,
                     hash(tuple(frozenset(self.valid_moves))),
                     self.board_score))

    def make_move(self, x: int, y: int) -> bool:
        if x == -1 and y == -1:  # skip the turn
            self.moves_skipped += 1
            self.current_player = -self.current_player
            self.valid_moves = self.get_valid_moves()
            self.moves_stack.append((-1, -1, None))
            return True
        if not self.is_valid_move(x, y):
            return False
        self.board[x, y] = self.current_player
        flipped = []
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            i, j = x + dx, y + dy
            while 0 <= i < 8 and 0 <= j < 8 and self.board[i, j] == -self.current_player:
                i, j = i + dx, j + dy
            if 0 <= i < 8 and 0 <= j < 8 and self.board[i, j] == self.current_player:
                i, j = i - dx, j - dy
                while i != x or j != y:
                    flipped.append((i, j))
                    self.board[i, j] = self.current_player
                    i, j = i - dx, j - dy
        self.current_player = -self.current_player
        self.turn_count += 1
        self.moves_skipped = 0
        self.valid_moves = self.get_valid_moves()
        self.moves_stack.append((x, y, flipped))
        self.empty_spaces.remove((x, y))
        return True

    def undo_move(self, rep: bool):
        if not self.moves_stack:
            return
        x, y, flipped = self.moves_stack.pop()
        if (x, y) == (-1, -1):  # if the last move was a skip
            self.moves_skipped = 0
            self.current_player = -self.current_player
            if rep:
                self.undo_move(True)
        else:  # normal move
            self.board[x, y] = 0
            for fx, fy in flipped:
                self.board[fx, fy] = self.current_player
            self.turn_count -= 1
            self.current_player = -self.current_player
            self.empty_spaces.add((x, y))
        self.valid_moves = self.get_valid_moves()

    def is_valid_move(self, x: int, y: int) -> bool:
        # Check if the position is empty
        if self.board[x, y] != 0:
            return False
        # Check if the move captures any opponent's discs

        for dx, dy in self.DIRECTIONS:
            if dx == dy == 0:
                continue
            i, j = x + dx, y + dy
            found_opponent = False
            while 0 <= i < 8 and 0 <= j < 8:
                if self.board[i, j] == 0:
                    break
                elif self.board[i, j] == -self.current_player:
                    found_opponent = True
                elif self.board[i, j] == self.current_player and found_opponent:
                    return True
                else:
                    break
                i += dx
                j += dy
        # If no opponent's discs are captured, the move is invalid
        return False

    def get_valid_moves(self) -> set:
        moves = {move for move in self.empty_spaces if self.is_valid_move(move[0], move[1])}
        return moves

    def is_game_over(self) -> bool:
        return self.turn_count >= 60 or self.moves_skipped >= 2

    @functools.lru_cache(maxsize=None)
    def get_sum(self) -> int:
        return int(np.sum(self.board))

    def get_piece_count(self, player: int) -> int:
        return np.count_nonzero(self.board == player)

    def get_winner(self) -> int:
        sum_ = self.get_sum()
        if sum_ > 0:
            return 1
        elif sum_ < 0:
            return -1
        else:
            return 0

    @functools.lru_cache(maxsize=None)
    def alpha_beta(self, depth: int, alpha: float, beta: float) -> Tuple[Tuple[int, int], float]:
        #print(self.alpha_beta.cache_info())
        # return a tuple of move, score
        if depth <= 0:
            return None, self.get_sum()  # if leaf node then return heuristic approximation
        if self.is_game_over():   # if leaf node then return game result
            win = self.get_winner()
            if win == 1:
                return None, 200
            elif win == -1:
                return None, -200
            else:
                return None, 0
        #self.valid_moves = self.get_valid_moves()
        valid_moves = self.valid_moves
        if len(valid_moves) == 0:  # if there are no moves then skip turn is the only move
            valid_moves.add((-1, -1))
        best_move = None
        if self.current_player == 1:  # maximizing player
            max_value = -float("inf")
            for move in valid_moves:
                self.make_move(move[0], move[1])
                _, value, = self.alpha_beta(depth - 1, alpha, beta)
                self.undo_move(False)
                if value > max_value:
                    max_value = value
                    best_move = move
                alpha = max(alpha, max_value)
                if max_value >= beta:
                    break  # beta cutoff
            return best_move, max_value
        else:  # minimizing player
            min_value = float("inf")
            for move in valid_moves:
                self.make_move(move[0], move[1])
                _, value, = self.alpha_beta(depth - 1, alpha, beta)
                self.undo_move(False)
                if value < min_value:
                    min_value = value
                    best_move = move
                beta = min(beta, value)
                if value <= alpha:
                    break  # alpha cutoff
            return best_move, min_value

    def get_best_move(self, depth):
        if self.valid_moves:  # make a random move
            best_move, score = self.alpha_beta(depth, -float("inf"), float("inf"))
            if best_move is not None:
                return best_move, score
        return None, None


def set_best_move(board: OthelloBoard, q: queue, depth: int, thread_numb: int):
    time.sleep(.001)
    res = board.__copy__().get_best_move(depth)
    q.put((res, thread_numb))


def draw_board(WIN, game_board: OthelloBoard, mx: int, my: int):
    WIN.fill(colors['green3'])
    for drx in range(8):
        for dry in range(8):
            dgx = SQUARE_SIZE * drx
            dgy = SQUARE_SIZE * dry

            if drx % 2 and dry % 2 or not drx % 2 and not dry % 2:
                drc = colors['green1']
            else:
                drc = colors['green2']
            pg.draw.rect(WIN, drc, (dgx, dgy, SQUARE_SIZE, SQUARE_SIZE))
            if game_board.board[drx, dry] == 1:
                pg.draw.circle(WIN, colors['black'], (dgx + HALF_SQUARE_SIZE, dgy + HALF_SQUARE_SIZE), HALF_SQUARE_SIZE - 5)
            if game_board.board[drx, dry] == -1:
                pg.draw.circle(WIN, colors['white'], (dgx + HALF_SQUARE_SIZE, dgy + HALF_SQUARE_SIZE), HALF_SQUARE_SIZE - 5)
    draw_legal_moves(WIN, game_board)
    if my < 8 * SQUARE_SIZE and mx < 8 * SQUARE_SIZE:  # mouse
        pg.draw.circle(WIN, colors['gray'], (mx // SQUARE_SIZE * SQUARE_SIZE + HALF_SQUARE_SIZE, my // SQUARE_SIZE * SQUARE_SIZE + HALF_SQUARE_SIZE), 10)
    else:
        pg.draw.circle(WIN, colors['gray'], (mx, my), 10)


def draw_legal_moves(WIN, game_board: OthelloBoard):
    legal_moves = game_board.valid_moves
    for drx, dry in legal_moves:
        dgx = SQUARE_SIZE * drx
        dgy = SQUARE_SIZE * dry
        if game_board.current_player == 1:
            c = colors['blue']
        else:
            c = colors['l-blue']
        pg.draw.circle(WIN, c, (dgx + HALF_SQUARE_SIZE, dgy + HALF_SQUARE_SIZE), HALF_SQUARE_SIZE - 22)


def draw_best_move(WIN, game_board: OthelloBoard, curr_best_move):
    best_move = curr_best_move
    if best_move is None:
        best_move = (-1, -1)
    drx, dry = best_move
    dgx = SQUARE_SIZE * drx
    dgy = SQUARE_SIZE * dry
    c = colors['yellow']
    pg.draw.circle(WIN, c, (dgx + HALF_SQUARE_SIZE, dgy + HALF_SQUARE_SIZE), HALF_SQUARE_SIZE - 30)


def draw_score(WIN, game_board: OthelloBoard, black_w: int, white_w: int, game_count: int, depth: int, current_best_score):
    black_p, white_p = game_board.get_piece_count(1), game_board.get_piece_count(-1)
    black = colors['black']
    scr = s_font.render(f"Black points: {str(black_p)}", True, black)
    WIN.blit(scr, (50, 610))
    scr = s_font.render(f"White points: {str(white_p)}", True, black)
    WIN.blit(scr, (410, 610))
    scr = s_font.render(f"Black wins: {str(black_w)}", True, black)
    WIN.blit(scr, (50, 630))
    scr = s_font.render(f"White wins: {str(white_w)}", True, black)
    WIN.blit(scr, (410, 630))
    board_score = game_board.get_sum()
    scr = s_font.render(f"Board score: {str(board_score)}", True, black)
    WIN.blit(scr, (210, 630))
    turn_count = game_board.turn_count
    scr = s_font.render(' ' + str(turn_count), True, black)
    WIN.blit(scr, (310, 610))
    if game_board.current_player == 1:
        t = "black"
        scr = s_font.render(f"Turn: {str(t)}", True, black)
        WIN.blit(scr, (210, 610))
    else:
        t = "white"
        scr = s_font.render(f"Turn: {str(t)}", True, black)
        WIN.blit(scr, (210, 610))

    s = current_best_score
    if s == 200:
        s = "Black Wins!"
    elif s == -200:
        s = "White Wins!"
    elif s == 0:
        s = 0  # ?"Draw!"
        if depth > 64:
            s = "Draw!"
    elif s is None:
        s = "Calculating..."
    scr = s_font.render(f"Game count: {game_count}", True, black)
    WIN.blit(scr, (50, 655))
    scr = s_font.render(f"Depth: {depth}", True, black)
    WIN.blit(scr, (210, 655))
    scr = s_font.render(f"Predicted score: {s}", True, black)
    WIN.blit(scr, (310, 655))


def main():
    try:
        #pg.init()  # initialize
        black_w, white_w = 0, 0
        current_best_move = (-1, -1)
        current_best_score = None
        game_count = 0
        WIN = pg.display.set_mode((600, 675))
        pg.display.set_caption("Othello - 0.2")
        game_running = True
        game_board = OthelloBoard()  # create board
        q = queue.Queue()  # thread
        depth = 1
        thread_numb = 0
        my_thread = threading.Thread(target=set_best_move, args=(game_board, q, depth, thread_numb))
        my_thread.start()

        while game_running:
            #depth = min(depth, 65)
            if not my_thread.is_alive():  # thread
                (game_board.best_move, game_board.board_score), numb = q.get()
                if numb == thread_numb:
                    current_best_move = game_board.best_move
                    current_best_score = game_board.board_score
                    depth += 1
                    thread_numb += 1
                    my_thread = threading.Thread(target=set_best_move, args=(game_board, q, depth, thread_numb))  # new thread
                    my_thread.start()
            time.sleep(.0001)
            mx, my = pg.mouse.get_pos()  # get mouse position
            keys = pg.key.get_pressed()  # get keys pressed
            if keys[pg.K_q]:  # button pressed
                if game_board.valid_moves:  # make a random move
                    random_move = random.choice(tuple(game_board.valid_moves))
                    game_board.make_move(random_move[0], random_move[1])
                    current_best_move = (-1, -1)
                    current_best_score = None
                pass
            if keys[pg.K_w]:  # button pressed
                if game_board.valid_moves:  # undo move
                    game_board.undo_move(True)
                    time.sleep(0.1)
                    current_best_move = (-1, -1)
                    current_best_score = None
                    depth = 1
                    thread_numb += 1
                    my_thread = threading.Thread(target=set_best_move, args=(game_board, q, depth, thread_numb))  # new thread
                    my_thread.start()
                pass
            for event in pg.event.get():  # check fo events
                if event.type == pg.KEYUP:  # button released
                    if keys[pg.K_o]:
                        pass
                    if keys[pg.K_q]:
                        depth = 1
                        thread_numb += 1
                        my_thread = threading.Thread(target=set_best_move, args=(game_board, q, depth, thread_numb))  # new thread
                        my_thread.start()
                    if keys[pg.K_r]:  # restart game
                        game_board = OthelloBoard()  # refresh board
                        current_best_move = (-1, -1)
                        current_best_score = None
                        depth = 1
                        thread_numb += 1
                        my_thread = threading.Thread(target=set_best_move, args=(game_board, q, depth, thread_numb))  # new thread
                        my_thread.start()
                    if keys[pg.K_p]: #
                        pass
                if event.type == pg.QUIT:  # quit
                    game_running = False
                if event.type == pg.MOUSEBUTTONUP:  # mouse click
                    if my < 8 * SQUARE_SIZE:
                        if (mx // SQUARE_SIZE, my // SQUARE_SIZE) in game_board.valid_moves:
                            game_board.make_move(mx // SQUARE_SIZE, my // SQUARE_SIZE)  # make a move
                            current_best_move = (-1, -1)
                            current_best_score = None
                            depth = 1
                            thread_numb += 1
                            my_thread = threading.Thread(target=set_best_move, args=(game_board, q, depth, thread_numb))  # new thread
                            my_thread.start()
            if not game_board.valid_moves:  # skip turn if there is no move
                game_board.make_move(-1, -1)
            if game_board.is_game_over(): #end game if needed
                if game_board.get_sum() > 0:
                    black_w += 1
                    game_count += 1
                elif game_board.get_sum() < 0:
                    white_w += 1
                    game_count += 1
                else:
                    game_count += 1
                game_board = OthelloBoard()  # refresh board
            depth = min(depth, 65)
            draw_board(WIN, game_board, mx, my)# draw board
            draw_score(WIN, game_board, black_w, white_w, game_count, depth, current_best_score)
            draw_best_move(WIN, game_board, current_best_move)
            pg.display.update()
        pg.quit()
    except KeyboardInterrupt:
        pg.quit()
        raise SystemExit
def test():
    pass
test()
main()
