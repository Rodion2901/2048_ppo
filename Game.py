import random
import math

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
                            reward += (math.log(new_row[i]*2, 2))//2
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
              return new_map, reward-1
       else:
              return m, -10

#функция ход вправо
def act_r(m):
       m_r = [r[::-1] for r in m]
       new_map, reward = move_marge(m_r)
       if new_map != m:
              place_rand_2(new_map)
              new_map = [r[::-1] for r in new_map]
              return new_map, reward-1
       else:
              return m, -10
#функция ход вверх
def act_u(m):
       trans_map = [list(col) for col in list(zip(*m))]
       new_map, reward = move_marge(trans_map)
       new_map = [list(r) for r in zip(*new_map)]
       if new_map != m:
              place_rand_2(new_map)
              return new_map, reward -1
       else:
              return m, -10
#функция ход вниз
def act_d(m):
       trans_map = [list(col) for col in list(zip(*m))]
       new_map, reward= move_marge(trans_map)
       new_map = [r[::-1] for r in new_map]
       new_map = [list(r) for r in zip(*new_map)]
       if new_map != m:
              place_rand_2(new_map)
              return new_map, reward -1
       else:
              return m, -10

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
       return True


class game_2048:
       def __init__(self):
              self.map = [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0]]
       def step(self, action):
              if action == 0:
                     self.map, reward = act_r(self.map)
              elif action == 1:
                     self.map, reward = act_l(self.map)
              elif action == 2:
                     self.map, reward = act_u(self.map)
              elif action == 3:
                     self.map, reward = act_d(self.map)
              if is_game_over(self.map):
                     return self.map, reward-50, True
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