"""
Multi-UAV grid navigation: hybrid BFS-based planner with PyTorch behavior cloning.

Run as a script: ``python multi_uav_grid_hybrid_planner.py``
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

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
                self.obstacles.add((random.randint(0, 19), random.randint(0, 19)))
            self.fixed_map = self.obstacles.copy()

        self.starts, self.goals = [], []
        for _ in range(NUM_UAVS):
            while True:
                s = (random.randint(0, 19), random.randint(0, 19))
                if s not in self.obstacles:
                    break
            while True:
                g = (random.randint(0, 19), random.randint(0, 19))
                if g not in self.obstacles and g != s:
                    break
            self.starts.append(s)
            self.goals.append(g)

        self.pos = self.starts.copy()
        return self.get_state()

    def get_state(self):
        state = []
        for i in range(NUM_UAVS):
            x, y = self.pos[i]
            gx, gy = self.goals[i]
            state.extend([x / 20, y / 20, gx / 20, gy / 20, (gx - x) / 20, (gy - y) / 20])
        return np.array(state)

    def step(self, actions):
        new_pos = []
        collision = False

        for i, a in enumerate(actions):
            x, y = self.pos[i]
            if a == 0:
                x -= 1
            elif a == 1:
                x += 1
            elif a == 2:
                y -= 1
            elif a == 3:
                y += 1

            x = np.clip(x, 0, 19)
            y = np.clip(y, 0, 19)
            new_pos.append((x, y))

        for i, p in enumerate(new_pos):
            if p in self.obstacles:
                collision = True
            for j in range(len(new_pos)):
                if i != j and p == new_pos[j]:
                    collision = True

        self.pos = new_pos

        rewards = []
        done = True

        for i in range(NUM_UAVS):
            dist = np.linalg.norm(np.array(self.pos[i]) - np.array(self.goals[i]))

            r = -0.01 * dist

            if dist <= 1:
                r += 200
            else:
                done = False

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
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_UAVS * ACTION_DIM),
        )

    def forward(self, s):
        x = self.net(s)
        x = x.view(-1, NUM_UAVS, ACTION_DIM)
        return torch.softmax(x, dim=2)


# ======================
# UTIL
# ======================
def next_pos(pos, action):
    x, y = pos
    if action == 0:
        x -= 1
    elif action == 1:
        x += 1
    elif action == 2:
        y -= 1
    elif action == 3:
        y += 1
    return (int(np.clip(x, 0, 19)), int(np.clip(y, 0, 19)))


def in_goal_zone(p, goal):
    return np.linalg.norm(np.array(p) - np.array(goal)) <= 1


def goal_zone_cells(goal):
    cells = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if in_goal_zone((x, y), goal):
                cells.append((x, y))
    return cells


_GOAL_ZONE_CACHE = {}


def get_goal_zone_set(goal):
    if goal not in _GOAL_ZONE_CACHE:
        _GOAL_ZONE_CACHE[goal] = set(goal_zone_cells(goal))
    return _GOAL_ZONE_CACHE[goal]


def is_safe(p, obstacles, occupied):
    return (p not in obstacles) and (p not in occupied)


def bfs_first_action(start, goal, obstacles, occupied_next):
    """
    Shortest path on grid to any cell in the goal zone (dist <= 1 from goal).
    Returns the first action (0-3) along a shortest path, or None if unreachable.
    """
    targets = get_goal_zone_set(goal)
    if start in targets:
        return None

    q = deque([start])
    parent = {start: None}
    action_from_parent = {start: None}
    # Must match next_pos: 0 left, 1 right, 2 down (y-1), 3 up (y+1)
    deltas = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]

    while q:
        cur = q.popleft()
        if cur in targets:
            first_a = None
            node = cur
            while parent[node] is not None:
                first_a = action_from_parent[node]
                node = parent[node]
            return first_a

        x, y = cur
        for dx, dy, a in deltas:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
                continue
            nxt = (nx, ny)
            if nxt in obstacles or nxt in occupied_next or nxt in parent:
                continue
            parent[nxt] = cur
            action_from_parent[nxt] = a
            q.append(nxt)

    return None


def greedy_fallback(env, i, occupied):
    x, y = env.pos[i]
    gx, gy = env.goals[i]
    actions = [0, 1, 2, 3]
    best = None
    best_score = 1e9

    for a in actions:
        p = next_pos((x, y), a)
        if not is_safe(p, env.obstacles, occupied):
            continue
        dist = np.linalg.norm(np.array(p) - np.array([gx, gy]))
        score = dist
        if score < best_score:
            best_score = score
            best = a

    if best is None:
        for a in actions:
            p = next_pos((x, y), a)
            if p not in env.obstacles:
                return a
        return 0
    return best


def best_action(env, i, occupied):
    x, y = env.pos[i]
    goal = env.goals[i]

    if in_goal_zone((x, y), goal):
        actions = [0, 1, 2, 3]
        random.shuffle(actions)
        for a in actions:
            p = next_pos((x, y), a)
            if is_safe(p, env.obstacles, occupied) and in_goal_zone(p, goal):
                return a
        for a in actions:
            p = next_pos((x, y), a)
            if is_safe(p, env.obstacles, occupied):
                return a
        return 0

    a = bfs_first_action((x, y), goal, env.obstacles, occupied)
    if a is not None:
        p = next_pos((x, y), a)
        if is_safe(p, env.obstacles, occupied):
            return a

    return greedy_fallback(env, i, occupied)


def hybrid_action(env):
    actions = [None] * NUM_UAVS
    occupied = set()

    order = sorted(
        range(NUM_UAVS),
        key=lambda i: np.linalg.norm(np.array(env.pos[i]) - np.array(env.goals[i])),
    )

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
optimizer = optim.Adam(actor.parameters(), lr=1e-3)

train_success = 0

for ep in range(5000):

    s = env.reset()
    success_flag = True

    for t in range(100):

        actions = hybrid_action(env)

        s2, r, d, collision = env.step(actions)

        st = torch.FloatTensor(s).unsqueeze(0)
        logits = actor.net(st).view(-1, NUM_UAVS, ACTION_DIM)
        target = torch.tensor(actions, dtype=torch.long).unsqueeze(0)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, ACTION_DIM), target.reshape(-1), reduction="mean"
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
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

    print(f"Episode {ep+1}, Success Rate: {(train_success / (ep + 1)) * 100:.2f}%")


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

print("\nFinal Test Success Rate:", success / TEST_EP * 100, "%")
