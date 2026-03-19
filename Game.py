import random
import math
import numpy as np

random.seed(42)
#pretty print
def pretty_print(m):
       print("----------")
       for i in range(4):
              print(*m[i])
       print("----------")
#функция которая будет возвращать индексы там, где 0
def zero_place(m):
       l = []
       for i in range(4):
              for j in range(4):
                     if m[i][j] == 0:
                            l.append([i,j])
       return l 
#функция которая помещает рандомно 2 двойки в map
def place_rand_2(m):
       l = zero_place(m)
       if len(l) == 0:
              return
       a = random.choice(l)
       m[a[0]][a[1]] = 2


#функция для передвижения и слияния
def move_marge(m):
       reward = 0
       new_m = []
       for row in m:
              #убираю нули
              new_row = [i for i in row if i !=0]
              #деляю слияние и меняю второе число на 0
              for i in range(len(new_row)-1):
                     if new_row[i] == new_row[i+1]:
                            new_row[i] *= 2
                            new_row[i+1] = 0
                            reward += (math.log(new_row[i]*2, 2))//4
              #убираю нули
              new_row = [i for i in new_row if i !=0]
              new_row = new_row + [0]*(4-len(new_row))
              new_m.append(new_row)
       return new_m, reward
       
#функция ход влево
def act_l(m):
       new_map, reward = move_marge(m)
       if new_map != m:
              place_rand_2(new_map)
              return new_map, reward-0.1
       else:
              return m, -1

#функция ход вправо
def act_r(m):
       m_r = [r[::-1] for r in m]
       new_map, reward = move_marge(m_r)
       if new_map != m_r:
              place_rand_2(new_map)
              new_map = [r[::-1] for r in new_map]
              return new_map, reward-0.1
       else:
              return m, -1
#функция ход вверх
def act_u(m):
       trans_map = [list(col) for col in list(zip(*m))]
       new_map, reward = move_marge(trans_map)
       new_map = [list(r) for r in zip(*new_map)]
       if new_map != m:
              place_rand_2(new_map)
              return new_map, reward -0.1
       else:
              return m, -1
#функция ход вниз
def act_d(m):
    trans_map = [list(col) for col in list(zip(*m))]
    trans_rev = [r[::-1] for r in trans_map]
    new_map_rev, reward = move_marge(trans_rev)
    new_map_trans = [r[::-1] for r in new_map_rev]
    new_map = [list(r) for r in zip(*new_map_trans)]
    if new_map != m:
        place_rand_2(new_map)
        return new_map, reward -0.1
    else:
        return m, -1
def is_plate_in_corner(map):
       reward = 0
       max_val = np.array(map).max()
       a = np.array(map).max()
       mid = [(1,1), (1,2), (2,1), (2,2)]
       is_in_mid = any(map[i][j] == a for i,j in mid)
       if is_in_mid:
              reward+= -3

       corner = [(0,0), (0,3), (3,0), (3,3)]

       is_in_corner = any(map[i][j] == max_val for i,j in corner)
       max_val = np.log2(max_val) * 2
       if is_in_corner:
              return reward + max_val
       else:
              return reward-max_val

def is_game_over(m):
       if len(zero_place(m)) != 0:
              return False
       for i in range(3):
              for j in range(3):
                     if (m[i][j] == m[i+1][j]) or (m[i][j] == m[i][j+1]):
                            return False
       for i in range(3):
              if m[3][i] == m[3][i+1]:
                     return False
       for i in range(3):
              if m[i][3] == m[i+1][3]:
                     return False
       return True
def monotonicity(map):
       reward = 0
       max_val = np.array(map).max()
       if max_val == map[0][0]:
              for i in range(3):
                     if map[0][i] == (map[0][i+1]/2):
                            reward += np.log2(map[0][i]) + np.log2(map[0][i+1])
                     else:
                            break
              if not reward:
                     return reward*2
              for i in range(3):
                     if map[i][0] == (map[i+1][0]/2):
                            reward += np.log2(map[i][0]) + np.log2(map[i+1][0])
                     else:
                            break
              if not reward:
                     return reward*2
       return -0.1
       

class game_2048:
       def __init__(self):
              self.map = [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]]
       def step(self, action):
              action = action[0]
              if action == 0:
                     self.map, reward = act_r(self.map)
              elif action == 1:
                     self.map, reward = act_l(self.map)
              elif action == 2:
                     self.map, reward = act_u(self.map)
              elif action == 3:
                     self.map, reward = act_d(self.map)
              reward += is_plate_in_corner(self.map)
              reward += len(zero_place(self.map)) * 0.5
              reward += monotonicity(self.map)
              if is_game_over(self.map):
                     return self.map, reward-10, True
              else:
                     return self.map, reward, False
       def reset(self):
              self.map = [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]]
              place_rand_2(self.map)
              place_rand_2(self.map)

              return self.map
       
# a = [
#        [0,0,0,0],
#        [0,2,0,0],
#        [0,0,0,0],
#        [0,0,2,0]
# ]
# is_plate_in_corner(map=a)