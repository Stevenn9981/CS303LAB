import numpy as np
from numpy import *


class Robot(object):

    DEPTH = 3
    next_point = [0, 0]
    cut_count = 0
    search_count = 0

    def __init__(self, _board, size, wait_list, color):
        self.board = _board
        self.size = size
        self.wait_list = wait_list
        self.color = color

    #计算所有空位点的价值
    def value_points(self, player, enemy, board):
        points = []
        aa = [0, 0, 0]
        cnt = 0
        for x in range(self.size):
            for y in range(self.size):
                hen_list = []  # 横排
                shu_list = []  # 竖排
                zuoxie_list = []  # 左下-右上斜对角线
                youxie_list = []  # 左上-右下斜对角线
                current_value = -1
                if self.board[x][y] == 0:
                    cnt += 1
                    if cnt == 1:
                        aa = [x, y, 0]
                    for m in range(x - 4, x + 5):
                        if 0 <= m < 15:
                            hen_list.append(board[m, y])
                        else:
                            hen_list.append(2)

                    for m in range(y - 4, y + 5):
                        if 0 <= m < 15:
                            shu_list.append(board[x, m])
                        else:
                            shu_list.append(2)

                    for m, n in zip(range(x - 4, x + 5), range(y - 4, y + 5)):
                        if 0 <= m < 15 and 0 <= n < 15:
                            zuoxie_list.append(board[m, n])
                        else:
                            zuoxie_list.append(2)

                    for m, n in zip(range(x - 4, x + 5), range(y + 4, y - 5, -1)):
                        if 0 <= m < 15 and 0 <= n < 15:
                            youxie_list.append(board[m, n])
                        else:
                            youxie_list.append(2)
                    me_value = self.calculate_value(player, enemy, hen_list, shu_list, zuoxie_list, youxie_list)
                    fan_value = self.calculate_value(enemy, player, hen_list, shu_list, zuoxie_list, youxie_list)
                    value = me_value + fan_value * 0.92
                    if value > 0:
                        points.append([x, y, value])
                    if value > current_value:
                        points.append([x, y, value])
                        self.wait_list.append([x, y])
                        current_value = value
        # print(points)
        return points


    #计算单个点的价值
    def value_point(self, point, player, enemy, board):
        x, y = point[0], point[1]
        hen_list = []  # 横排
        shu_list = []  # 竖排
        zuoxie_list = []  # 左下-右上斜对角线
        youxie_list = []  # 左上-右下斜对角线
        value = 0
        if self.board[x][y] == 0:
            for m in range(x - 4, x + 5):
                if 0 <= m < 15:
                    hen_list.append(board[m, y])
                else:
                    hen_list.append(2)

            for m in range(y - 4, y + 5):
                if 0 <= m < 15:
                    shu_list.append(board[x, m])
                else:
                    shu_list.append(2)

            for m, n in zip(range(x - 4, x + 5), range(y - 4, y + 5)):
                if 0 <= m < 15 and 0 <= n < 15:
                    zuoxie_list.append(board[m, n])
                else:
                    zuoxie_list.append(2)

            for m, n in zip(range(x - 4, x + 5), range(y + 4, y - 5, -1)):
                if 0 <= m < 15 and 0 <= n < 15:
                    youxie_list.append(board[m, n])
                else:
                    youxie_list.append(2)
            me_value = self.calculate_value(player, enemy, hen_list, shu_list, zuoxie_list, youxie_list)
            fan_value = self.calculate_value(enemy, player, hen_list, shu_list, zuoxie_list, youxie_list)
            value = me_value + fan_value * 0.92
        return value

    # 基本搜索算法
    def max_value_point(self, player, enemy):
        points = self.value_points(player, enemy, self.board)
        flag = -1
        max_point = []
        for p in points:
            if p[2] > flag:
                max_point = p
                flag = p[2]
        return max_point

    def calculate_value(self, player, enemy, hen_list, shu_list, zuoxie_list, youxie_list):
        flag = 0
        flag += self.eval_value(player, hen_list) + self.eval_value(player, shu_list) + self.eval_value(player,
                                                                                                        zuoxie_list) + self.eval_value(
            player, youxie_list)

        return flag

    @staticmethod
    def eval_value(player, checklist):
        checklist[4] = player
        huo_wu = [player,player,player,player,player]
        huo_si = [0, player, player, player, player, 0]
        si_si_A = [player, player, player, player, 0]
        si_si_B = [player, player, player, 0, player]
        si_si_C = [player, player, 0, player, player]
        huo_san = [0, player, player, player, 0]
        si_san_A = [player, player, player, 0, 0]
        si_san_B = [0, player, 0, player, player, 0]
        si_san_C = [player, 0, 0, player, player]
        si_san_D = [player, 0, player, 0, player]
        huo_er = [0, 0, 0, player, player, 0, 0, 0]
        si_er_A = [player, player, 0, 0, 0]
        si_er_B = [0, 0, player, 0, player, 0, 0]
        si_er_C = [0, player, 0, 0, player, 0]

        for i in range(5):
            if checklist[i:i + 5] == huo_wu:
                checklist[4] = 0
                return 5000000

        for i in range(4):
            if checklist[i:i + 6] == huo_si:
                checklist[4] = 0
                return 300000

        for i in range(5):
            if checklist[i:i + 5] == si_si_A:
                checklist[4] = 0
                return 2500

        for i in range(5):
            if checklist[i:i + 5] == si_si_B:
                checklist[4] = 0
                return 3000

        for i in range(5):
            if checklist[i:i + 5] == si_si_C:
                checklist[4] = 0
                return 2600

        for i in range(5):
            if checklist[i:i + 5] == huo_san:
                checklist[4] = 0
                return 3000

        for i in range(5):
            if checklist[i:i + 5] == si_san_A:
                checklist[4] = 0
                return 500

        for i in range(4):
            if checklist[i:i + 6] == si_san_B:
                checklist[4] = 0
                return 800

        for i in range(5):
            if checklist[i:i + 5] == si_san_C:
                checklist[4] = 0
                return 600

        for i in range(5):
            if checklist[i:i + 5] == si_san_D:
                checklist[4] = 0
                return 550

        for i in range(2):
            if checklist[i:i + 8] == huo_er:
                checklist[4] = 0
                return 650

        for i in range(5):
            if checklist[i:i + 5] == si_er_A:
                checklist[4] = 0
                return 150

        for i in range(3):
            if checklist[i:i + 7] == si_er_B:
                checklist[4] = 0
                return 250

        for i in range(4):
            if checklist[i:i + 6] == si_er_C:
                checklist[4] = 0
                return 200

        checklist[4] = 0
        return 0

    #使用α-β剪枝
    def ai(self, player, enemy):
        self.minmax(player, enemy, self.DEPTH, -999999999, 999999999)
        return self.next_point

    # 极大极小值算法搜索 alpha + beta剪枝
    def minmax(self, player, enemy, depth, alpha, beta):
        if depth == 0:
            return self.max_value_point(player, enemy)[2]

        blank_list = []

        for m in range(self.size):
            for n in range(self.size):
                if self.board[m,n] == 0:
                    blank_list.append([m,n])

        for next_step in blank_list:

            if not self.has_neightbor(next_step):
                continue

            self.board[next_step[0], next_step[1]] = player
            value = - self.minmax( enemy, player, depth - 1, -beta, -alpha)
            self.board[next_step[0], next_step[1]] = 0

            if value > alpha:
                if depth == self.DEPTH:
                    self.next_point[0] = next_step[0]
                    self.next_point[1] = next_step[1]

                if value >= beta:
                    return beta
                alpha = value

        return alpha

    def has_neightbor(self, pt):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= pt[0] + i < self.size and 0 <= pt[1] + j < self.size and self.board[pt[0] + i, pt[1] + j]:
                    return True
        return False


COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0


# don't change the class name
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need add your decision into your candidate_list. System will get the end of your candidate_list as your decision .
        self.candidate_list = []

    # # If your are the first, this function will be used.
    # def first_chess(self, chessboard):
    #     assert self.color == COLOR_BLACK
    #     self.candidate_list.clear()
    #     # ==================================================================
    #     # Here you can put your first piece
    #     # for example, you can put your piece on sun（天元）
    #     self.candidate_list.append((self.chessboard_size // 2, self.chessboard_size // 2))
    #     chessboard[self.candidate_list[-1][0], self.candidate_list[-1][0]] = self.color

    # The input is current chessboard.
    def go(self, chessboard):
        robot = Robot(chessboard, self.chessboard_size, self.candidate_list, self.color)
        # Clear candidate_list
        self.candidate_list.clear()
        new_pos = (0, 0)
        # ==================================================================
        # To write your algorithm here
        # Here is the simplest sample:Random decision
        if (chessboard == zeros((self.chessboard_size, self.chessboard_size))).all() and self.color == COLOR_BLACK:
            new_pos = [self.chessboard_size // 2, self.chessboard_size // 2]
            self.candidate_list.append(new_pos)
        else:
            new_pos = robot.ai(self.color, - self.color)
        # ==============Find new pos========================================
        # Make sure that the position of your decision in chess board is empty.
        # If not, return error.
        assert chessboard[new_pos[0], new_pos[1]] == 0
        chessboard[new_pos[0], new_pos[1]] = self.color
        # Add your decision into candidate_list, Records the chess board
        self.candidate_list.append(new_pos)

# broad = zeros((15, 15), dtype=int)
chessboard = np.zeros((15, 15))
ai1 = AI(15, -1, 1000)
ai2 = AI(15, 1, 1000)
a = 10
# print(chessboard)
while a > 0:
    ai1.go(chessboard)
    print(ai1.candidate_list[-1])
    ai2.go(chessboard)
    a = a - 1
    print(chessboard)
    print()

# chessboard_size = 15
#
# chessboard = np.zeros((chessboard_size, chessboard_size), dtype=np.int)
# chessboard[1, 0:2] = -1
# chessboard[2:4, 2] = -1
# chessboard[1, 6:8] = 1
# chessboard[2:4, 8] = 1
#
# print(chessboard)
# ai1 = AI(15, -1, 1000)
# ai1.go(chessboard)
# print(ai1.candidate_list[-1])
# print(chessboard)