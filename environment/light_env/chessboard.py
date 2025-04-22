from environment.light_env.common import *
from environment.lookup_tables import Winner
from collections import defaultdict

class L_Chessboard:

    def __init__(self, init=None, verbose = False):
        self.height = 10
        self.verbose = verbose
        self.width = 9
        self.board = [['.' for col in range(self.width)] for row in range(self.height)]
        self.steps = 0
        self._legal_moves = None
        self.turn = RED
        self.winner = None
        self.repetition_dict = defaultdict(int)
        if init is None or init == '':
            self.assign_fen(None)
        else:
            self.parse_init(init)

    def _update(self):
        self._fen = None
        self._legal_moves = None
        self.steps += 1
        if self.steps % 2 == 0:
            self.turn = RED
        else:
            self.turn = BLACK
        
        # Count current board situation for repetition draw detection
        fen = self.FENboard().split(' ')[0]  # Not include player side, only consider the situation
        self.repetition_dict[fen] += 1

    def parse_init(self, init):
        # Parse 32-char board string with 2-digit coordinates per piece
        pieces = 'rnbakabnrccpppppRNBAKABNRCCPPPPP'
        position = [init[i:i+2] for i in range(len(init)) if i % 2 == 0]
        for pos, piece in zip(position, pieces):
            if pos != '99':
                x, y = int(pos[0]), 9 - int(pos[1])
                self.board[y][x] = piece

    def assign_fen(self, fen):
        if fen is None:
            # Simplified Fen (Situation): Each side have one cannon, two
            fen = "2r1k1r2/9/4c4/4p4/9/4P4/9/4C4/9/2R1K1R2 r - - 0 1"
        x = 0
        y = 0
        for k in range(0, len(fen)):
            ch = fen[k]
            if ch == ' ':
                if (fen[k+1] == 'b'):
                    self.turn = BLACK
                break
            if ch == '/':
                x = 0
                y += 1
            elif ch >= '1' and ch <= '9':
                for i in range(int(ch)):
                    self.board[y][x] = '.'
                    x = x + 1
            else:
                self.board[y][x] = ch
                x = x + 1

    def FENboard(self):
        def swapcase(a):
            if a.isalpha():
                a = replace_dict[a]
                return a.lower() if a.isupper() else a.upper()
            return a

        c = 0
        fen = ''
        for i in range(self.height - 1, -1, -1):
            c = 0
            for j in range(self.width):
                if self.board[i][j] == '.':
                    c = c + 1
                else:
                    if c > 0:
                        fen = fen + str(c)
                    fen = fen + swapcase(self.board[i][j])
                    c = 0
            if c > 0:
                fen = fen + str(c)
            if i > 0:
                fen = fen + '/'
        if self.turn is RED:
            fen += ' r'
        else:
            fen += ' b'
        fen += ' - - 0 1'
        return fen

    def fliped_FENboard(self):
        # Return flipped version of the board in FEN
        fen = self.FENboard()
        foo = fen.split(' ')
        rows = foo[0].split('/')
        def swapcase(a):
            if a.isalpha():
                return a.lower() if a.isupper() else a.upper()
            return a
        def swapall(aa):
            return "".join([swapcase(a) for a in aa])

        return "/".join([swapall(reversed(row)) for row in reversed(rows)]) \
            + " " + foo[1] \
            + " " + foo[2] \
            + " " + foo[3] + " " + foo[4] + " " + foo[5]

    @property
    def is_red_turn(self):
        return self.turn == RED

    @property
    def screen(self):
        return self.board
    
    @property
    def done(self):
        return self.is_end()
    

    def step(self, action):
        # If turn on verbose mode, print at your demand
        if self.verbose:
            print("[Step] Executing move:", action)
            for row in self.board[::-1]:
                print(row)

        target_piece = self.move_action_str(action)
        # self.steps += 1


        # If capture the general/king, the game terminates
        if target_piece == 'K':
            self.winner = Winner.black
            return self, -1, True, {}
        elif target_piece == 'k':
            self.winner = Winner.red
            return self, 1, True, {}

        # Otherwise, follow the regular stream
        done = self.is_end()
        reward = 0
        if done:
            if self.winner == Winner.red:
                reward = 1
            elif self.winner == Winner.black:
                reward = -1
            else:
                reward = 0  # draw
        return self, reward, done, {}


    def reset(self):
        self.assign_fen(None)
        self.steps = 0
        self._legal_moves = None
        self.winner = None
        self.repetition_dict.clear()
        self._update()

    def legal_moves(self):
        if self._legal_moves is not None:
            return self._legal_moves

        _legal_moves = []
        for y in range(self.height):
            for x in range(self.width):
                ch = self.board[y][x]
                if (self.turn == RED and ch.isupper()):
                    continue
                if (self.turn == BLACK and ch.islower()):
                    continue

                if ch in mov_dir:
                    for d in mov_dir[ch]:
                        x_ = x + d[0]
                        y_ = y + d[1]
                        if not self._can_move(x_, y_):
                            continue
                        elif ch == 'p' and y < 5 and x_ != x:  # red pawn can't move sideways before crossing river
                            continue
                        elif ch == 'P' and y > 4 and x_ != x:  # black pawn can't move sideways before crossing river
                            continue
                        elif ch in ['n', 'N', 'b', 'B']:  # knight or bishop block check
                            if self.board[y + int(d[1]/2)][x + int(d[0]/2)] != '.':
                                continue
                            if ch == 'b' and y_ > 4:
                                continue
                            if ch == 'B' and y_ < 5:
                                continue
                        elif ch in ['k', 'a'] and (x_ < 3 or x_ > 5 or y_ > 2):
                            continue
                        elif ch in ['K', 'A'] and (x_ < 3 or x_ > 5 or y_ < 7):
                            continue
                        _legal_moves.append(move_to_str(x, y, x_, y_))

                    # King-to-King rule
                    if ch == 'k' and self.turn == RED:
                        d, u = self._y_board_from(x, y)
                        if u < self.height and self.board[u][x] == 'K':
                            _legal_moves.append(move_to_str(x, y, x, u))
                    elif ch == 'K' and self.turn == BLACK:
                        d, u = self._y_board_from(x, y)
                        if d >= 0 and self.board[d][x] == 'k':
                            _legal_moves.append(move_to_str(x, y, x, d))

                elif ch in ['r', 'R', 'c', 'C']:
                    l, r = self._x_board_from(x, y)
                    d, u = self._y_board_from(x, y)

                    # shift (not kill pieces)
                    for x_ in range(l + 1, x):
                        _legal_moves.append(move_to_str(x, y, x_, y))
                    for x_ in range(x + 1, r):
                        _legal_moves.append(move_to_str(x, y, x_, y))
                    for y_ in range(d + 1, y):
                        _legal_moves.append(move_to_str(x, y, x, y_))
                    for y_ in range(y + 1, u):
                        _legal_moves.append(move_to_str(x, y, x, y_))

                    if ch in ['r', 'R']:
                        if self._can_move(l, y): _legal_moves.append(move_to_str(x, y, l, y))
                        if self._can_move(r, y): _legal_moves.append(move_to_str(x, y, r, y))
                        if self._can_move(x, d): _legal_moves.append(move_to_str(x, y, x, d))
                        if self._can_move(x, u): _legal_moves.append(move_to_str(x, y, x, u))

                    elif ch in ['c', 'C']:
                        # cannon kill pieces
                        l_, _ = self._x_board_from(l, y)
                        _, r_ = self._x_board_from(r, y)
                        d_, _ = self._y_board_from(x, d)
                        _, u_ = self._y_board_from(x, u)

                        for xx, yy in [(l_, y), (r_, y), (x, d_), (x, u_)]:
                            if 0 <= xx < self.width and 0 <= yy < self.height:
                                target = self.board[yy][xx]
                                if target != '.' and not self._is_same_side(xx, yy):
                                    _legal_moves.append(move_to_str(x, y, xx, yy))

        # Forced capture king (If exist)
        def move_str_to_coords(m):
            return int(m[0]), int(m[1]), int(m[2]), int(m[3])

        king_capture_moves = []
        for move in _legal_moves:
            sx, sy, dx, dy = move_str_to_coords(move)
            target = self.board[dy][dx]
            if (self.turn == RED and target == 'K') or (self.turn == BLACK and target == 'k'):
                king_capture_moves.append(move)

        if king_capture_moves:
            self._legal_moves = king_capture_moves
        else:
            self._legal_moves = _legal_moves

        return self._legal_moves

    def is_legal(self, mov):
        return mov.uci in self.legal_moves
    
    
    
    def is_end(self):
        MAX_STEPS = 100

        # Step 1: Check whether the both sides' kings exist
        red_k_exist = any('k' in row for row in self.board)
        black_k_exist = any('K' in row for row in self.board)

        if not red_k_exist:
            self.winner = Winner.black
            return True
        elif not black_k_exist:
            self.winner = Winner.red
            return True

        # Step 2: check king face to king
        red_k, black_k = [-1, -1], [-1, -1]
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j] == 'k':
                    red_k = [i, j]
                elif self.board[i][j] == 'K':
                    black_k = [i, j]

        if red_k[1] == black_k[1] and red_k[0] != -1 and black_k[0] != -1:
            has_block = False
            for row in range(red_k[0] + 1, black_k[0]):
                if self.board[row][red_k[1]] != '.':
                    has_block = True
                    break
            if not has_block:
                self.winner = Winner.red if self.turn == RED else Winner.black
                return True

        # Step 3: check if there is any legal move
        legal = self.legal_moves()
        if not legal:
            self.winner = Winner.black if self.turn == RED else Winner.red
            return True

        # If current legal moves include capturing the king, we don't announe the victory first
        # We need to announe after capturing
        for move in legal:
            dx, dy = int(move[2]), int(move[3])
            target = self.board[dy][dx]
            if (self.turn == RED and target == 'K') or (self.turn == BLACK and target == 'k'):
                return False

        # Step 4: Reach max steps, announce 'draw'
        if self.steps >= MAX_STEPS:
            self.winner = Winner.draw
            return True

        return self.winner is not None


    def print_to_cl(self):
        for i in range(9, -1, -1):
            print(self.board[i])

    def move_action_str(self, uci):
        sx, sy, dx, dy = int(uci[0]), int(uci[1]), int(uci[2]), int(uci[3])
        target_piece = self.board[dy][dx]  # Get killed piece
        mov = Move(uci)
        self.push(mov)
        return target_piece  # return target piece

    def push(self, mov):
        self.board[mov.n[1]][mov.n[0]] = self.board[mov.p[1]][mov.p[0]]
        self.board[mov.p[1]][mov.p[0]] = '.'
        self._update()


    def _is_same_side(self,x,y):
        if self.turn == RED and self.board[y][x].islower():
            return True
        if self.turn == BLACK and self.board[y][x].isupper():
            return True

    def _can_move(self,x,y): # basically check the move
        if x < 0 or x > self.width-1:
            return False
        if y < 0 or y > self.height-1:
            return False
        if self._is_same_side(x,y):
            return False
        return True

    def _x_board_from(self,x,y):
        l = x-1
        r = x+1
        while l > -1 and self.board[y][l] == '.':
            l = l-1
        while r < self.width and self.board[y][r] == '.':
            r = r+1
        return l,r

    def _y_board_from(self,x,y):
        d = y-1
        u = y+1
        while d > -1 and self.board[d][x] == '.':
            d = d-1
        while u < self.height and self.board[u][x] == '.':
            u = u+1
        return d,u

    def result(self, claim_draw=True) -> str:
        if self.winner == Winner.red:
            return '1-0'
        elif self.winner == Winner.black:
            return '0-1'
        elif self.winner == Winner.draw:
            return '1/2-1/2'
        else:
            return '*'
        

    def clear_chessmans_moving_list(self):
        return

    def calc_chessmans_moving_list(self):
        return

    def save_record(self, filename):
        return

    def parse_WXF_move(self, wxf):
        '''
        red is upper, black is lower alphabet
        '''
        p = self.swapcase(wxf[0])
        col = wxf[1]
        mov = wxf[2]
        dest_col = wxf[3]
        src_row, src_col = self.find_row(p, col)
        if mov == '.' or mov == '=':
            # move horizontally
            dest_row = src_row
            if p.islower():
                dest_col = int(dest_col) - 1
            else:
                dest_col = self.width - int(dest_col)
        else:
            if p == 'h' or p == 'H' or p == 'e' or p == 'E' or p == 'a' or p == 'A':
                if p.islower():
                    dest_col = int(dest_col) - 1
                else:
                    dest_col = self.width - int(dest_col)

                if p == 'h' or p == 'H':
                    # for house/knight
                    step = 1 if abs(dest_col - src_col) == 2 else 2
                elif p == 'e' or p == 'E':
                    # for elephant/bishop
                    step = 2
                else:
                    # for advisor
                    step = 1 
                if mov == '+' and p.islower() or mov == '-' and p.isupper():
                    dest_row = src_row + step
                else:
                    dest_row = src_row - step
            else:
                # move vertically
                step = int(dest_col)
                if mov == '+' and p.islower() or mov == '-' and p.isupper():
                    dest_row = src_row + step
                else:
                    dest_row = src_row - step
                dest_col = src_col
        return move_to_str(src_col, src_row, dest_col, dest_row)

    def find_row(self, piece, col):
        if piece == 'h' or piece == 'H':
            piece = 'n' if piece == 'h' else 'N'
        if piece == 'e' or piece == 'E':
            piece = 'b' if piece == 'e' else 'B'
        column = 0
        row = -1
        if col.isdigit():
            if piece.isupper():
                column = self.width - int(col)
            else:
                column = int(col) - 1
            for i in range(self.height):
                if self.board[i][int(column)] == piece:
                    row = i
                    break
        else:
            first_row = -1
            second_row = -1
            column = -1
            for j in range(self.width):
                column = -1
                for i in range(self.height):
                    if self.board[i][j] == piece:
                        if column == -1:
                            column = j
                            first_row = i
                        else:
                            if column == j:
                                second_row = i
                                break
                            else:
                                column = j
                                first_row = second_row = -1
                if first_row != -1 and second_row != -1:
                    break
            if (piece.islower() and col == '+') or (piece.isupper() and col == '-'):
                row = second_row
            else:
                row = first_row
        return row, column

    def swapcase(self, a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a



if __name__ == '__main__': # test
    board = L_Chessboard()
    print(board.legal_moves)