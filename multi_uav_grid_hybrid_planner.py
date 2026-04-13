import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# ======================
# PARAMETERS
# ======================
GRID_SIZE = 20
NUM_UAVS = 3
ACTION_DIM = 4
STATE_DIM = NUM_UAVS * 6

# ======================
# ENVIRONMENT
# ======================
class GridEnv:
    def __init__(self):
        self.num_obstacles = 40
        self.fixed_map = None
        self.reset()

    def reset(self, fixed=False):
        if fixed and self.fixed_map is not None:
            self.obstacles = self.fixed_map
        else:
            self.obstacles = set()
            while len(self.obstacles) < self.num_obstacles:
                self.obstacles.add((random.randint(0,19), random.randint(0,19)))
            self.fixed_map = self.obstacles.copy()

        self.starts, self.goals = [], []
        for _ in range(NUM_UAVS):
            while True:
                s = (random.randint(0,19), random.randint(0,19))
                if s not in self.obstacles:
                    break
            while True:
                g = (random.randint(0,19), random.randint(0,19))
                if g not in self.obstacles and g != s:
                    break
            self.starts.append(s)
            self.goals.append(g)

        self.pos = self.starts.copy()
        return self.get_state()

    def get_state(self):
        state=[]
        for i in range(NUM_UAVS):
            x,y = self.pos[i]
            gx,gy = self.goals[i]
            state.extend([x/20,y/20,gx/20,gy/20,(gx-x)/20,(gy-y)/20])
        return np.array(state)

    def step(self, actions):
        new_pos=[]
        collision=False

        for i,a in enumerate(actions):
            x,y=self.pos[i]
            if a==0:x-=1
            elif a==1:x+=1
            elif a==2:y-=1
            elif a==3:y+=1

            x=np.clip(x,0,19)
            y=np.clip(y,0,19)
            new_pos.append((x,y))

        # collision detection
        for i,p in enumerate(new_pos):
            if p in self.obstacles:
                collision=True
            for j in range(len(new_pos)):
                if i!=j and p==new_pos[j]:
                    collision=True

        self.pos=new_pos

        rewards=[]
        done=True

        for i in range(NUM_UAVS):
            dist=np.linalg.norm(np.array(self.pos[i])-np.array(self.goals[i]))

            r = -0.01*dist

            if dist<=1:
                r += 200
            else:
                done=False

            if collision:
                r -= 50

            rewards.append(r)

        return self.get_state(), sum(rewards), done, collision


# ======================
# NETWORK
# ======================
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(STATE_DIM,256),
            nn.ReLU(),
            nn.Linear(256,NUM_UAVS*ACTION_DIM)
        )

    def forward(self,s):
        x=self.net(s)
        x=x.view(-1,NUM_UAVS,ACTION_DIM)
        return torch.softmax(x,dim=2)


# ======================
# UTIL
# ======================
def next_pos(pos, action):
    x,y = pos
    if action==0:x-=1
    elif action==1:x+=1
    elif action==2:y-=1
    elif action==3:y+=1
    return (np.clip(x,0,19), np.clip(y,0,19))


def is_safe(p, obstacles, occupied):
    return (p not in obstacles) and (p not in occupied)


def best_action(env, i, occupied):
    """
    Multi-step heuristic with safety
    """
    x,y = env.pos[i]
    gx,gy = env.goals[i]

    actions = [0,1,2,3]
    best = None
    best_score = 1e9

    for a in actions:
        p = next_pos((x,y), a)

        if not is_safe(p, env.obstacles, occupied):
            continue

        # distance + small penalty
        dist = np.linalg.norm(np.array(p) - np.array([gx,gy]))
        score = dist

        if score < best_score:
            best_score = score
            best = a

    # fallback if stuck
    if best is None:
        for a in actions:
            p = next_pos((x,y), a)
            if p not in env.obstacles:
                return a
        return 0

    return best


def hybrid_action(env):
    """
    Strong deterministic hybrid controller
    """
    actions = [None]*NUM_UAVS
    occupied = set()

    # priority: closest first
    order = sorted(range(NUM_UAVS),
                   key=lambda i: np.linalg.norm(np.array(env.pos[i]) - np.array(env.goals[i])))

    for i in order:
        a = best_action(env, i, occupied)
        actions[i] = a
        occupied.add(next_pos(env.pos[i], a))

    return actions


# ======================
# TRAIN (lightweight, stable)
# ======================
env = GridEnv()
actor = Actor()
optimizer = optim.Adam(actor.parameters(), lr=3e-4)

train_success = 0

for ep in range(5000):

    s = env.reset()
    success_flag = True

    for t in range(100):

        actions = hybrid_action(env)

        s2, r, d, collision = env.step(actions)

        # simple imitation-style update
        st = torch.FloatTensor(s).unsqueeze(0)
        target = torch.zeros((NUM_UAVS, ACTION_DIM))

        for i,a in enumerate(actions):
            target[i][a] = 1

        loss = ((actor(st)[0] - target)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        s = s2

        if collision:
            success_flag = False

        if d:
            break

    for i in range(NUM_UAVS):
        if np.linalg.norm(np.array(env.pos[i]) - np.array(env.goals[i])) > 1:
            success_flag = False

    if success_flag:
        train_success += 1

    print(f"Episode {ep+1}, Success Rate: {(train_success/(ep+1))*100:.2f}%")


# ======================
# TEST
# ======================
success = 0
TEST_EP = 1000

for _ in range(TEST_EP):

    s = env.reset(fixed=True)
    collision = False

    for t in range(100):

        actions = hybrid_action(env)

        s, r, d, collision = env.step(actions)

        if d:
            break

    ok = True
    for i in range(NUM_UAVS):
        if np.linalg.norm(np.array(env.pos[i]) - np.array(env.goals[i])) > 1:
            ok = False

    if ok and not collision:
        success += 1

print("\nFinal Test Success Rate:", success/TEST_EP*100, "%")