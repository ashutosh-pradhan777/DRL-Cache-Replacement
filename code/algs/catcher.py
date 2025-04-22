import numpy as np
from .lib.cacheop import CacheOp
from .lib.dequedict import DequeDict
from .lib.heapdict import HeapDict
from .lib.pollutionator import Pollutionator
from .lib.visualizinator import Visualizinator
from .catcher_agent import CatcherDDPG

class Catcher:
    def __init__(self, cache_size, warmup=500, period_length=1000, **kwargs):
        self.cache = {}
        self.capacity = cache_size
        self.lru = DequeDict()
        self.lfu = HeapDict()
        self.ddpg = CatcherDDPG(state_dim=10)
        self.window = []
        self.recent_state = None
        self.visual = Visualizinator()
        self.pollution = Pollutionator(period_length)
        self.tick = 0
        self.train_every = kwargs.get("train_every", 100)
        self.warmup = warmup

    def request(self, lba, ts=None):
        self.tick += 1
        self.pollution.setUnique(lba)
        self.window.append([lba])
        if len(self.window) > 10:
            self.window.pop(0)
        state = self.window.copy()
        while len(state) < 10:
            state.insert(0, [0]) 
        state = np.array(state, dtype=np.float32)


        if lba in self.cache:
            self.pollution.incrementUniqueCount()
            self.lru[lba] = self.tick
            if lba in self.lfu:
                self.lfu[lba] += 1
            else:
                self.lfu[lba] = 1  # Initialize if not present
            self.visual.addWindow(self.tick, 'hits', 1)
            return CacheOp.HIT, None


        prob = self.ddpg.act(state)
        if len(self.cache) >= self.capacity:
            evicted = self.lru.popFirst() if np.random.rand() < prob else self.lfu.popMin()
            if evicted in self.cache:
                del self.cache[evicted]
                del self.lru[evicted]
                del self.lfu[evicted]
        else:
            evicted = None

        self.cache[lba] = 1
        self.lru[lba] = self.tick
        self.lfu[lba] = 1

        if self.recent_state is not None:
            r = 1.0 if evicted is not None else -1.0
            self.ddpg.buffer.add(self.recent_state, [prob], [r], state)
            if self.tick > self.warmup and self.tick % self.train_every == 0:
                self.ddpg.train()


        self.recent_state = state
        self.visual.addWindow(self.tick, 'misses', 1)
        self.pollution.update(self.tick)
        return CacheOp.INSERT, evicted
