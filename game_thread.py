import pygame
from pygame.locals import *
from sys import exit
import threading
import time
import reinforce_learning as rl
import random
import math

random.seed(time.time())


class GameThread(threading.Thread):

    screen_width = 640
    screen_height = 560

    line_num = 15
    width = 36

    piece_width = 18

    board_offset = [screen_width / 40, screen_height / 40]

    screen = pygame.display.set_mode((screen_width, screen_height), 0, 32)

    color_dict = {-1: (255, 255, 255), 1: (0, 0, 0)}

    now_color = 1
    chess_board = [[0 for col in range(15)] for row in range(15)]

    history = []

    step_num = 0

    def __init__(self, thread_id):
        threading.Thread.__init__(self)
        self.id = thread_id
        self.game_count = 0
        self.training_mode = False
        self.current_episode = []
        self.use_mcts_in_play = True
        self.mcts_simulations = 100

    def loop(self):
        self.display()

        mouse_pos = [-1, -1]

        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    exit()
                # if event.type == MOUSEBUTTONUP:
                #     if event.button == 1:
                #         mouse_pos2 = self.get_xy(event.pos)
                #         if mouse_pos2 == mouse_pos and self.chess_board[mouse_pos[0]][mouse_pos[1]] == 0:
                #             self.chess_board[mouse_pos[0]][mouse_pos[1]] = self.now_color
                #             if self.is_win(mouse_pos[0], mouse_pos[1], self.now_color):
                #                 self.add_train_data()
                #                 self.init_board()
                #             self.history.append(self.copy_self())
                #             self.now_color = -self.now_color
                #             self.step_num += 1
                # if event.type == MOUSEBUTTONDOWN:
                #     if event.button == 1:
                #         mouse_pos = self.get_xy(event.pos)
            self.display()
            time.sleep(0.2)

    def run(self):
        time.sleep(1)
        self.training_mode = True
        # always start from latest saved model if available
        try:
            if rl.load_latest_model():
                print("loaded latest model for training")
        except Exception as e:
            print(f"load latest failed: {e}")

        while True:
            try:
                self.play_training_game()
            except Exception as exc:
                print(f"Training game error: {exc}")

    def place_pieces(self, x, y):
        self.chess_board[x][y] = self.now_color
        self.history.append(self.copy_self())
        self.step_num += 1
        if self.is_win(x, y, self.now_color):
            self.handle_game_end(self.now_color)
            return True
        if self.step_num >= self.line_num * self.line_num:
            self.handle_game_end(0)
            return True
        self.now_color = -self.now_color
        return False

    def handle_game_end(self, winner_color):
        self.game_count += 1
        print(f"Game {self.game_count} finished. Winner: {winner_color}")
        if self.training_mode:
            try:
                rl.log_game_result(winner_color)
            except Exception:
                pass
            rl.record_episode(self.current_episode, winner_color)
            rl.train()
            try:
                rl.save_model(step=rl._opt_step)
            except Exception:
                rl.save_model()
        else:
            # during human play pause shortly to show final board
            time.sleep(1)
        self.current_episode = []
        self.init_board()

    def legal_moves(self):
        moves = []
        for i in range(self.line_num):
            for j in range(self.line_num):
                if self.chess_board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def play_training_game(self):
        self.current_episode = []
        self.init_board()
        try:
            rl.load_latest_model()
        except Exception as load_err:
            print(f"Warning: could not reload latest model before game: {load_err}")
        print(f"Starting game {self.game_count + 1}")
        finished = False
        while not finished:
            board_state = self.copy_self()
            state_input = self.to_input(board_state)
            legal_moves = self.legal_moves()
            action_idx, log_prob, value, _ = rl.select_action(state_input, legal_moves)
            x = action_idx // self.line_num
            y = action_idx % self.line_num
            if (x, y) not in legal_moves:
                x, y = random.choice(legal_moves)
                action_idx = x * self.line_num + y
                log_prob = 0.0
            self.current_episode.append({
                'state': state_input,
                'action': action_idx,
                'log_prob': log_prob,
                'value': value,
                'color': self.now_color
            })
            finished = self.place_pieces(x, y)

    def ai_move(self, deterministic=True):
        # ensure we are using the freshest weights saved during training
        rl.load_latest_model()
        legal_moves = self.legal_moves()
        if not legal_moves:
            return
        board_state = self.copy_self()
        state_input = self.to_input(board_state)
        if deterministic and self.use_mcts_in_play:
            move = self._mcts_select()
            if move is None:
                action_idx, _, _, _ = rl.select_action(state_input, legal_moves, deterministic=True)
                move = (action_idx // self.line_num, action_idx % self.line_num)
        else:
            action_idx, _, _, _ = rl.select_action(state_input, legal_moves, deterministic=deterministic)
            move = (action_idx // self.line_num, action_idx % self.line_num)
        if move not in legal_moves:
            move = random.choice(legal_moves)
        self.place_pieces(move[0], move[1])

    def _board_to_tuple(self, board):
        return tuple(tuple(row) for row in board)

    def _board_clone(self, board):
        return [row[:] for row in board]

    def _board_apply(self, board, move, color):
        new_board = self._board_clone(board)
        new_board[move[0]][move[1]] = color
        return new_board

    def _board_legal_moves(self, board):
        moves = []
        for i in range(self.line_num):
            for j in range(self.line_num):
                if board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def _board_is_full(self, board):
        for row in board:
            for cell in row:
                if cell == 0:
                    return False
        return True

    def _board_check_win(self, board, move, color):
        if move is None:
            return False
        return self._check_direction(board, move, color, (1, 0)) or \
            self._check_direction(board, move, color, (0, 1)) or \
            self._check_direction(board, move, color, (1, 1)) or \
            self._check_direction(board, move, color, (1, -1))

    def _check_direction(self, board, move, color, delta):
        count = 1
        for direction in (-1, 1):
            dx, dy = delta
            x, y = move
            while True:
                x += dx * direction
                y += dy * direction
                if x < 0 or x >= self.line_num or y < 0 or y >= self.line_num:
                    break
                if board[x][y] == color:
                    count += 1
                else:
                    break
        return count >= 5

    def _board_to_input(self, board, player):
        transformed = [[board[i][j] * player for j in range(self.line_num)] for i in range(self.line_num)]
        return self.to_input(transformed)

    def _mcts_select(self):
        simulations = self.mcts_simulations
        root_board = self._board_clone(self.chess_board)
        root_player = self.now_color
        if self._board_is_full(root_board):
            return None

        N = {}
        W = {}
        P = {}
        children = {}
        c_puct = 1.5

        def expand(board, player):
            node_key = (self._board_to_tuple(board), player)
            legal = self._board_legal_moves(board)
            if not legal:
                P[node_key] = {}
                children[node_key] = []
                N[node_key] = 1
                W[node_key] = 0.0
                return 0.0
            state_input = self._board_to_input(board, player)
            _, _, value, probs = rl.select_action(state_input, legal, deterministic=True)
            priors = {}
            total = 0.0
            for mv in legal:
                idx = mv[0] * self.line_num + mv[1]
                prior = float(probs[idx])
                priors[mv] = prior
                total += prior
            if total <= 0:
                weight = 1.0 / len(legal)
                for mv in priors:
                    priors[mv] = weight
            else:
                for mv in priors:
                    priors[mv] /= total
            P[node_key] = priors
            children[node_key] = list(legal)
            N[node_key] = 1
            W[node_key] = value
            return value

        def simulate(board, player):
            node_key = (self._board_to_tuple(board), player)
            legal = self._board_legal_moves(board)
            if not legal:
                return 0.0
            if node_key not in P:
                return expand(board, player)
            parent_visits = N.get(node_key, 1)
            best_move = None
            best_child_board = None
            best_score = -float('inf')
            for mv in children[node_key]:
                child_board = self._board_apply(board, mv, player)
                child_key = (self._board_to_tuple(child_board), -player)
                q = 0.0
                if child_key in W and child_key in N and N[child_key] > 0:
                    q = W[child_key] / N[child_key]
                u = c_puct * P[node_key].get(mv, 0.0) * math.sqrt(parent_visits) / (1 + N.get(child_key, 0))
                score = q + u
                if score > best_score:
                    best_score = score
                    best_move = mv
                    best_child_board = child_board
            if best_move is None:
                return expand(board, player)
            winner = self._board_check_win(best_child_board, best_move, player)
            if winner:
                value = 1.0
            elif self._board_is_full(best_child_board):
                value = 0.0
            else:
                value = -simulate(best_child_board, -player)
            N[node_key] = N.get(node_key, 0) + 1
            W[node_key] = W.get(node_key, 0.0) + value
            return value

        for _ in range(simulations):
            simulate(root_board, root_player)

        best_move = None
        best_visits = -1
        for mv in self.legal_moves():
            child_board = self._board_apply(root_board, mv, root_player)
            child_key = (self._board_to_tuple(child_board), -root_player)
            visits = N.get(child_key, 0)
            if visits > best_visits:
                best_visits = visits
                best_move = mv
        if best_move is None:
            legal_moves = self.legal_moves()
            if not legal_moves:
                return None
            best_move = random.choice(legal_moves)
        return best_move

    def copy_self(self):
        board_copy = [[0 for col in range(self.line_num)] for row in range(self.line_num)]
        length = len(self.chess_board)
        side = self.now_color
        for i in range(length):
            for j in range(length):
                board_copy[i][j] = side*self.chess_board[i][j]

        return board_copy

    def get_xy(self, pos):
        x = (pos[0] - self.board_offset[0] + self.width/2) / self.width
        y = (pos[1] - self.board_offset[1] + self.width/2) / self.width
        xy = (int(x), int(y))
        return xy

    def draw_piece(self, chess_color, pos):
        x = self.board_offset[0] + pos[0]*self.width
        y = self.board_offset[1] + pos[1]*self.width
        xy = (int(x), int(y))
        color_num = self.color_dict[chess_color]
        pygame.draw.circle(self.screen, color_num, xy, self.piece_width)

    def draw_board(self):
        self.screen.fill((100, 255, 100))
        for i in range(0, self.line_num):
            pygame.draw.line(self.screen, (0, 0, 0), (self.board_offset[0] + i * self.width, self.board_offset[1]),
                             (self.board_offset[0] + i * self.width,
                              self.board_offset[1] + (self.line_num - 1) * self.width))
            pygame.draw.line(self.screen, (0, 0, 0), (self.board_offset[0], self.board_offset[1] + i * self.width),
                             (self.board_offset[0] + (self.line_num - 1) * self.width,
                              self.board_offset[1] + i * self.width))

    def init_board(self):
        length = len(self.chess_board)
        for i in range(length):
            for j in range(length):
                self.chess_board[i][j] = 0

        self.history = []
        self.now_color = 1
        self.step_num = 0

    def display(self):
        pygame.init()
        self.draw_board()
        for i in range(len(self.chess_board)):
            for j in range(len(self.chess_board[i])):
                if self.chess_board[i][j] != 0:
                    self.draw_piece(self.chess_board[i][j], (i, j))
        pygame.display.update()

    def to_input(self, board):
        c = [[[0.0 for col in range(2)] for col in range(self.line_num)] for row in range(self.line_num)]
        length = len(board)
        for i in range(length):
            for j in range(length):
                if board[i][j] == 1:
                    c[i][j][0] = 1.0
                elif board[i][j] == -1:
                    c[i][j][1] = 1.0
        return c

    def is_win(self, i, j, color):
        length = len(self.chess_board)
        a = 5
        count = 1
        for x in range(1, a):
            tx = i - x
            ty = j
            if tx < 0 or tx >= length:
                break
            if self.chess_board[tx][ty] == color:
                count += 1
            else:
                break

        for x in range(1, a):
            tx = i + x
            ty = j
            if tx < 0 or tx >= length:
                break
            if self.chess_board[tx][ty] == color:
                count += 1
            else:
                break

        if count >= 5:
            return True

        count = 1
        for x in range(1, a):
            tx = i - x
            ty = j - x
            if tx < 0 or tx >= length:
                break
            if ty < 0 or ty >= length:
                break
            if self.chess_board[tx][ty] == color:
                count += 1
            else:
                break

        for x in range(1, a):
            tx = i + x
            ty = j + x
            if tx < 0 or tx >= length:
                break
            if ty < 0 or ty >= length:
                break
            if self.chess_board[tx][ty] == color:
                count += 1
            else:
                break

        if count >= 5:
            return True

        count = 1
        for x in range(1, a):
            tx = i
            ty = j - x
            if ty < 0 or ty >= length:
                break
            if self.chess_board[tx][ty] == color:
                count += 1
            else:
                break

        for x in range(1, a):
            tx = i
            ty = j + x
            if ty < 0 or ty >= length:
                break
            if self.chess_board[tx][ty] == color:
                count += 1
            else:
                break

        if count >= 5:
            return True

        count = 1
        for x in range(1, a):
            tx = i - x
            ty = j + x
            if tx < 0 or tx >= length:
                break
            if ty < 0 or ty >= length:
                break
            if self.chess_board[tx][ty] == color:
                count += 1
            else:
                break

        for x in range(1, a):
            tx = i + x
            ty = j - x
            if tx < 0 or tx >= length:
                break
            if ty < 0 or ty >= length:
                break
            if self.chess_board[tx][ty] == color:
                count += 1
            else:
                break

        if count >= 5:
            return True
        return False

    def loop_human_vs_ai(self):
        """Interactive play: human (black, 1) vs latest PPO model (white, -1)."""
        self.training_mode = False
        loaded = rl.load_latest_model()
        if not loaded:
            print("No saved model found. Playing with freshly initialized weights.")
        self.init_board()
        human_color = 1
        self.now_color = 1
        self.display()
        mouse_pos = [-1, -1]
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    exit()
                if event.type == KEYDOWN and event.key == K_r:
                    print("Resetting board")
                    self.init_board()
                if event.type == MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = self.get_xy(event.pos)
                if event.type == MOUSEBUTTONUP and event.button == 1:
                    pos = self.get_xy(event.pos)
                    if pos == mouse_pos:
                        x, y = pos
                        if 0 <= x < self.line_num and 0 <= y < self.line_num and self.chess_board[x][y] == 0 and self.now_color == human_color:
                            finished = self.place_pieces(x, y)
                            if not finished:
                                self.ai_move(deterministic=True)
            self.display()
            time.sleep(0.05)
