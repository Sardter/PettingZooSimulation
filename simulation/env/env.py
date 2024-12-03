import random
from enum import Enum

from pettingzoo import AECEnv
import math

class ItemType(Enum):
    EMPTY = 0
    FOOD = 1
    OBSTACLE = 2
    AGENTS = 3
    

class AgentRole(Enum):
    VILLAGER = 1
    THIEF = 2
    
class AgentAction(Enum):
    MOVE = 1
    GATHER = 2
    TRADE = 3
    EAT = 4
    STEAL = 5
    EXECUTE = 6


class GridItem:
    id: int
    pos_x: int
    post_y: int
    item_type: ItemType
    item: any

    def __init__(self, id: int, item_type: ItemType, pos_x: int, pos_y: int, item) -> None:
        self.id = id
        self.item_type = item_type
        self.pos_x = pos_x
        self.post_y = pos_y
        self.item = item


class CacheItem:
    pos_x: int
    pos_y: int
    
    def __init__(self, x, y) -> None:
        self.pos_x = x
        self.pos_y = y


class Agent:
    hunger: float # [0, 1]
    inventory: int
    reputation: float # [-1, 1]
    role: AgentRole



class SimulationEnviornment(AECEnv):
    metadata = {
        "name": "custom_environment_v0",
    }
    
    __length: int
    __grid: list[list[GridItem]]
    __obstacle_generation_rate: float
    __food_generation_rate: float
    
    __grid_food_count: int
    __grid_obstacle_count: int
    __grid_agents_count: int
    __time: int

    __agents_cahce: list[CacheItem]
    __food_cahce: list[CacheItem]
    __obstacle_cache: list[CacheItem]

    def __init__(self, 
        length: int, 
        obstacle_generation_rate: float, 
        food_generation_rate) -> None:
        
        assert 0 <= obstacle_generation_rate + food_generation_rate < 1
        
        self.__length = length
        self.__obstacle_generation_rate = obstacle_generation_rate
        self.__food_generation_rate = food_generation_rate
        
        self.__grid_agents_count = 0
        self.__grid_obstacle_count = 0
        self.__grid_food_count = 0
        self.__time = 0
        
        self.__grid = [[self.__init_generate_grid_item(i, j) for j in range(self.__length)] for i in range(self.__length)]
        
    
    def get_food_quantity(self) -> int:
        return self.__grid_food_count
    
    def is_cell_passable(self, x: int, y: int) -> bool:
        return self.__grid[x][y].item_type != ItemType.OBSTACLE
    
    
    def __generate_food(self, pos_x: int, pos_y: int):
        item_type = ItemType.FOOD
        id = self.__grid_food_count
        self.__grid_food_count += 1
        item = 1
        self.__food_cahce.append(CacheItem(pos_x, pos_y))
        return GridItem(id=id, item_type=item_type, pos_x=pos_x, pos_y=pos_y, item=item)
    
    
    def __generate_obstacle(self, pos_x: int, pos_y: int):
        item_type = ItemType.OBSTACLE
        id = self.__grid_obstacle_count
        self.__grid_obstacle_count += 1
        item = None
        self.__obstacle_cache.append(CacheItem(pos_x, pos_y))
        return GridItem(id=id, item_type=item_type, pos_x=pos_x, pos_y=pos_y, item=item)
    
    def __generate_empty(self, pos_x: int, pos_y: int):
        id = -1
        item_type = ItemType.EMPTY
        item = None
        
        return GridItem(id=id, item_type=item_type, pos_x=pos_x, pos_y=pos_y, item=item)
    
    
    def __init_generate_grid_item(self, pos_x: int, pos_y: int) -> GridItem:
        dice = random.uniform(0, 1)
        
        if dice < self.__obstacle_generation_rate:
            return self.__generate_obstacle(pos_x, pos_y)
        elif dice < self.__obstacle_generation_rate + self.__food_generation_rate:
            return self.__generate_food(pos_x, pos_y)
        
        return self.__generate_empty(pos_x, pos_y)
    
    
    def __regenerate_food(self) -> None:
        for x in range(self.__length):
            for y in range(self.__length):
                item_type = self.__grid[x][y].item_type
                dice = random.uniform(0, 1)
                
                if dice >= self.__food_generation_rate:
                    continue
                
                if item_type == ItemType.EMPTY:
                    self.__grid[x][y] = self.__generate_food(x, y)
                elif item_type == ItemType.FOOD:
                    self.__grid[x][y].item += 1
                    
    
    def __calculate_observation_spaces(self) -> None:
        for agent_pos in self.__agents_cahce:
            x, y = agent_pos.pos_x, agent_pos.pos_y
            agents = self.__grid[x][y]
            
            for every_other_agent_pos in self.__agents_cahce:
                xj, yj = every_other_agent_pos.pos_x, every_other_agent_pos.pos_y
                if xj == x and yj == y: continue
                other_agents = self.__grid[xj][yj]
                
                euclidiean_distance = math.sqrt((xj - x) ** 2 + (yj - y) ** 2)
            
            for food_pos in self.__food_cahce:
                xj, yj = food_pos.pos_x, food_pos.pos_y
                food = self.__grid[xj][yj]
                
                euclidiean_distance = math.sqrt((xj - x) ** 2 + (yj - y) ** 2)
                
                

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]