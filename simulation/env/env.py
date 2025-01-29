import random

from pettingzoo import AECEnv
from gymnasium import spaces
import math


class GridItem:
    pos_x: int
    pos_y: int

    def __init__(self, pos_x: int, pos_y: int) -> None:
        self.pos_x = pos_x
        self.pos_y = pos_y


class Obstacle(GridItem):
    ...


class CacheItem:
    pos_x: int
    pos_y: int
    
    def __init__(self, x, y) -> None:
        self.pos_x = x
        self.pos_y = y


class Agent:
    agent_id: str
    hunger: float # [0, 1]
    inventory: int
    reputation: float # [-1, 1]
    
    def __init__(
        self,
        *,
        agent_id: str = "agent_id",
        hunger: float = 1,
        inventory: int = 0,
        reputation: float = 0
    ):
        self.agent_id = agent_id
        self.hunger = hunger
        self.inventory = inventory
        self.reputation = reputation
    

class Villager(Agent):
    ...


class Thief(Agent):
    ...


class NotObstacle(GridItem):
    food_count: int
    villagers: list[Villager]
    thieves: list[Thief]
    
    def __init__(
        self, 
        pos_x, 
        pos_y,
        food_count: int = 0,
        villagers: list[Villager] = [],
        theives: list[Thief] = []
    ):
        self.food_count = food_count
        self.villagers = villagers
        self.thieves = theives
        super().__init__(pos_x, pos_y)


class SimulationEnviornment(AECEnv):
    metadata = {
        "name": "custom_environment_v0",
    }
    
    __length: int
    __grid: list[list[GridItem]]
    __food_generation_rate: float
    __max_agent_sight_distance: int
    __max_agent_inventory_limit: int
    __max_agent_gather_limit: int
    __max_move_limit: int
    
    __grid_food_count: int
    __grid_obstacle_count: int
    __grid_agents_count: int
    __time: int

    __agents_cahce: list[CacheItem]
    __food_cahce: list[CacheItem]
    __obstacle_cache: list[CacheItem]

    def __init__(
        self,
        *,
        length: int, 
        food_generation_rate: float,
        max_agent_sight_distance: int,
        max_agent_inventory_limit: int,
        max_agent_gather_limit: int,
        max_agent_move_limit: int,
        villager_count: int,
        thief_count: int,
        obstacle_count: int,
    ) -> None:
        
        assert 0 <= food_generation_rate < 1
        
        self.__length = length
        self.__food_generation_rate = food_generation_rate
        self.__max_agent_sight_distance = max_agent_sight_distance
        self.__max_agent_inventory_limit = max_agent_inventory_limit
        self.__max_agent_gather_limit = max_agent_gather_limit
        self.__max_agent_move_limit = max_agent_move_limit
        
        self.__grid_agents_count = villager_count + thief_count
        self.__grid_obstacle_count = obstacle_count
        self.__grid_food_count = int(round((self.__length ** 2) * self.__food_generation_rate))
        self.__time = 0
        
        self.__possible_villagers = [f"villager_{i}" for i in range(villager_count)]
        self.__possible_thieves = [f"thief_{i}" for i in range(thief_count)]
        
        self.possible_agents = self.__possible_villagers + self.__possible_thieves        
    
    
    def get_food_quantity(self) -> int:
        return self.__grid_food_count
    
    
    def is_cell_passable(self, x: int, y: int) -> bool:
        return not isinstance(self.__grid[x][y].item_type, Obstacle)
    
    
    def reset(self, seed=None, options=None):
        self.__generate_grid()

    
    def __generate_obstacle(self, pos_x: int, pos_y: int):
        self.__obstacle_cache.append(CacheItem(pos_x, pos_y))
        return Obstacle(pos_x=pos_x, pos_y=pos_y)
    
    
    def __generate_not_obstacle(self, pos_x: int, pos_y: int):
        return NotObstacle(pos_x=pos_x, pos_y=pos_y)
    
    
    def __generate_grid(self):
        self.__grid = [[self.__generate_not_obstacle(x, y) for y in range(self.__length)] for x in range(self.__length)]
        availible_positions = [(x, y) for y in range(self.__length) for x in range(self.__length)]
        self.action_spaces = {}
        
        for _ in range(self.__grid_obstacle_count):
            rand_x, rand_y = random.choice(availible_positions)
            self.__grid[rand_x][rand_y] = self.__generate_obstacle(rand_x, rand_y)
            del availible_positions[rand_x * self.__length + rand_y]
        
        
        for _ in range(self.__grid_food_count):
            rand_x, rand_y = random.choice(availible_positions)
            self.__grid[rand_x][rand_y].food_count += 1
        
        
        for villager_name in self.__possible_villagers:
            rand_x, rand_y = random.choice(availible_positions)
            villager = Villager(agent_id=villager_name)
            action_space = spaces.Dict({
                "action_type": spaces.Discrete(5),
                "move_amount": spaces.Discrete(self.__max_agent_move_limit),
                "gather_amount": spaces.Discrete(self.__max_agent_gather_limit),
                "trade_item": spaces.Discrete(self.__max_agent_inventory_limit),
                "eat_item": spaces.Discrete(self.__max_agent_inventory_limit),
                "execute_target": spaces.Discrete(self.__grid_agents_count),
            })
            
            self.__grid[rand_x][rand_y].villagers.append(villager)
            self.action_spaces[villager_name] = action_space
        
        
        for thief_name in self.__possible_thieves:
            rand_x, rand_y = random.choice(availible_positions)
            thief = Thief(agent_id=thief_name)
            action_space = spaces.Dict({
                "action_type": spaces.Discrete(5),
                "move_amount": spaces.Discrete(self.__max_agent_move_limit),
                "gather_amount": spaces.Discrete(self.__max_agent_gather_limit),
                "trade_item": spaces.Discrete(self.__max_agent_inventory_limit),
                "eat_item": spaces.Discrete(self.__max_agent_inventory_limit),
                "steal_target": spaces.Discrete(self.__grid_agents_count),
            })
            
            self.__grid[rand_x][rand_y].thieves.append(thief)
            self.action_spaces[thief_name] = action_space
            
    
    
    
    def __bresenham_line(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        current_x, current_y = x0, y0

        while True:
            yield current_x, current_y
            if current_x == x1 and current_y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                current_x += sx
            if e2 < dx:
                err += dx
                current_y += sy
    
    
    def __can_see(self, agents: GridItem, other: GridItem) -> bool:
        agents_x, agents_y = agents.pos_x, agents.pos_y
        other_x, other_y = other.pos_x, other.pos_y
        
        euclidiean_distance = math.sqrt((agents_x - other_x) ** 2 + (agents_y - other_y) ** 2)
        if euclidiean_distance > self.__max_agent_sight_distance:
            return False
        
        for x, y in self.__bresenham_line(agents_x, agents_y, other_x, other_y):
            looking_for_an_obstacle = isinstance(other, Obstacle) and x == other_x and y == other_y
            if self.is_cell_passable(x, y) and not looking_for_an_obstacle:
                return False
        
        return True
     
    
    def __calculate_observation_spaces(self) -> None:
        observations_spaces = dict()
        for agent_pos in self.__agents_cahce:
            x, y = agent_pos.pos_x, agent_pos.pos_y
            agents_pos: GridItem = self.__grid[x][y]
            agents: list[Agent] = agents_pos.item
            
            for agent in agents:
                
                for every_other_agent_pos in self.__agents_cahce:
                    xj, yj = every_other_agent_pos.pos_x, every_other_agent_pos.pos_y
                    if xj == x and yj == y: continue
                    other_agents = self.__grid[xj][yj]
                    
                    
                
                for food_pos in self.__food_cahce:
                    xj, yj = food_pos.pos_x, food_pos.pos_y
                    food = self.__grid[xj][yj]


    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]