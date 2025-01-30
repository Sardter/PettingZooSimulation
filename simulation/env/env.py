import random
from typing import Tuple
from copy import copy

from pettingzoo import AECEnv
from gymnasium import spaces
import numpy as np
import math


class GridItem:
    pos_x: int
    pos_y: int

    def __init__(self, pos_x: int, pos_y: int) -> None:
        self.pos_x = pos_x
        self.pos_y = pos_y


class Obstacle(GridItem):
    ...


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
    villagers: dict[str, Villager]
    thieves: dict[str, Thief]
    
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

    __agents_cahce: dict[str, Tuple[int, int]]
    __food_cahce: dict[int, Tuple[int, int, int]]
    __obstacle_cache: dict[int, Tuple[int, int]]

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
        self.agents = copy(self.possible_agents)

    
    def __generate_obstacle(self, pos_x: int, pos_y: int):
        self.__obstacle_cache[pos_x * self.__length + pos_y] = (pos_x, pos_y)
        return Obstacle(pos_x=pos_x, pos_y=pos_y)
    
    
    def __generate_not_obstacle(self, pos_x: int, pos_y: int):
        return NotObstacle(pos_x=pos_x, pos_y=pos_y)
    
    
    def __generate_villager_action_space(self) -> spaces.Dict:
        return spaces.Dict({
            "action_type": spaces.Discrete(6),
            "move_amount": spaces.Discrete(self.__max_agent_move_limit),
            "gather_amount": spaces.Discrete(self.__max_agent_gather_limit),
            "trade_item": spaces.Discrete(self.__max_agent_inventory_limit),
            "eat_item": spaces.Discrete(self.__max_agent_inventory_limit),
            "execute_target": spaces.Discrete(self.__grid_agents_count),
        })
    
    
    def __generate_thief_action_space(self) -> spaces.Dict:
        return spaces.Dict({
            "action_type": spaces.Discrete(6),
            "move_amount": spaces.Discrete(self.__max_agent_move_limit),
            "gather_amount": spaces.Discrete(self.__max_agent_gather_limit),
            "trade_item": spaces.Discrete(self.__max_agent_inventory_limit),
            "eat_item": spaces.Discrete(self.__max_agent_inventory_limit),
            "steal_target": spaces.Discrete(self.__grid_agents_count),
        })
    
    def __generate_agent_observation_space(self, villager_name: str) -> spaces.Dict:
        return spaces.Dict({
            "position": spaces.Box(low=0, high=self.__length, shape=(2,), dtype=np.int32),
            "hunger": spaces.Box(0, 1),
            "inventory": spaces.Discrete(self.__max_agent_inventory_limit),
            "visible_agents": spaces.Dict({
                other_agent: spaces.Dict({
                    "position": spaces.Box(low=0, high=self.__length, shape=(2,), dtype=np.int32),
                    "visible": spaces.Discrete(2),
                })
                for other_agent in self.possible_agents if other_agent != villager_name
            }),
            "agents_reputations": spaces.Dict({
                other_agent: spaces.Dict({
                    "repution": spaces.Box(low=-1, high=1),
                })
                for other_agent in self.possible_agents if other_agent != villager_name
            }),
            "visible_foods": spaces.Dict({
                food_id: spaces.Dict({
                    "position": spaces.Box(low=0, high=self.__length, shape=(2,), dtype=np.int32),
                    "visible": spaces.Discrete(2),
                    "count": spaces.Discrete(10)
                })
                for food_id in self.__food_cahce
            }),
            "movable_spaces": spaces.Box(low=0, high=self.__max_agent_move_limit, 
                    shape=(self.__max_agent_move_limit * 2 + 1, self.__max_agent_move_limit * 2 + 1), dtype=np.int32)
        })
    
    
    def __generate_food(self, x: int, y: int):
        self.__grid[x][y].food_count += 1
        self.__food_cahce[x * self.__length + y] = (x, y, self.__grid[x][y].food_count)
    
    
    def __generate_villager(self, x: int, y: int, villager_name: str):
        self.__agents_cahce[villager_name] = (x, y)
            
        villager = Villager(agent_id=villager_name)
        action_space = self.__generate_villager_action_space()
        observation_space = self.__generate_agent_observation_space(villager_name)
        
        self.__grid[x][y].villagers[villager_name] = villager
        self.action_spaces[villager_name] = action_space
        self.observation_spaces[villager_name] = observation_space
    
    
    def __generate_thief(self, x: int, y: int, thief_name: str):
        self.__agents_cahce[thief_name] = (x, y)
        thief = Thief(agent_id=thief_name)
        action_space = self.__generate_thief_action_space()
        observation_space = self.__generate_agent_observation_space(thief_name)
        
        self.__grid[x][y].thieves[thief_name] = thief
        self.action_spaces[thief_name] = action_space
        self.observation_spaces[thief_name] = observation_space
    
    
    def __generate_grid(self):
        self.__grid = [[self.__generate_not_obstacle(x, y) for y in range(self.__length)] for x in range(self.__length)]
        availible_positions = [(x, y) for y in range(self.__length) for x in range(self.__length)]
        self.action_spaces = dict()
        self.observation_spaces = dict()
        
        for _ in range(self.__grid_obstacle_count):
            rand_x, rand_y = random.choice(availible_positions)
            self.__grid[rand_x][rand_y] = self.__generate_obstacle(rand_x, rand_y)
            del availible_positions[rand_x * self.__length + rand_y]
        
        
        for _ in range(self.__grid_food_count):
            rand_x, rand_y = random.choice(availible_positions)
            self.__generate_food(rand_x, rand_y)
        
        
        for villager_name in self.__possible_villagers:
            rand_x, rand_y = random.choice(availible_positions)
            self.__generate_villager(rand_x, rand_y, villager_name)
        
        
        for thief_name in self.__possible_thieves:
            rand_x, rand_y = random.choice(availible_positions)
            self.__generate_thief(rand_x, rand_y, thief_name)

    
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
    
    
    def __can_see(self, agent: GridItem, other: GridItem) -> bool:
        agent_x, agent_y = agent.pos_x, agent.pos_y
        other_x, other_y = other.pos_x, other.pos_y
        
        euclidiean_distance = math.sqrt((agent_x - other_x) ** 2 + (agent_y - other_y) ** 2)
        if euclidiean_distance > self.__max_agent_sight_distance:
            return False
        
        for x, y in self.__bresenham_line(agent_x, agent_y, other_x, other_y):
            looking_for_an_obstacle = isinstance(other, Obstacle) and x == other_x and y == other_y
            if self.is_cell_passable(x, y) and not looking_for_an_obstacle:
                return False
        
        return True

    
    def observe(self, agent_name: str):
        x, y = self.__agents_cahce[agent_name]
        agent: Agent = self.__grid[x][y].agents[agent_name]
        
        visible_agents = {
            other_name: {
                "visible": self.__can_see(self.__grid[x][y], self.__grid[other_x][other_y]), 
                "position": [other_x, other_y]
            }
            for other_name, (other_x, other_y) in self.__agents_cahce.items()
            if other_name != agent_name
        }
        
        reputations = {
            other_name: {
                "reputation": self.__grid[x][y].agents[other_name].reputation,
            }
            for other_name in self.__agents_cahce.keys()
            if other_name != agent_name
        }
        
        visible_foods = {
            food_key: {
                "visible": self.__can_see(self.__grid[x][y], self.__grid[food_x][food_y]), 
                "position": [food_x, food_y],
                "count": food_count
            }
            for food_key, (food_x, food_y, food_count) in self.__food_cahce.items()
        }
        
        movable_spaces = np.full((self.__max_agent_move_limit * 2 + 1, self.__max_agent_move_limit * 2 + 1), -1)
        
        for i in range(max(x - self.__max_agent_move_limit, 0), min(x + self.__max_agent_move_limit, self.__length)):
            for j in range(max(y - self.__max_agent_move_limit, 0), min(y + self.__max_agent_move_limit, self.__length)):
                movable_spaces[i, j] = round(math.sqrt((i - x) ** 2 + (j - y) ** 2)) if self.is_cell_passable(i, j) else -1
            
        return {
            "position": [x, y],
            "hunger": agent.hunger,
            "inventory": agent.inventory,
            "visible_agents": {other_name: {
                "position": data["position"] if data['visible'] else [-1, -1], 
                "visible":data['visible'] } 
                for other_name, data in visible_agents.items()
            },
            "agents_reputations": reputations,
            "visible_foods": {food_key: {
                "position": data["position"] if data['visible'] else [-1, -1], 
                "visible":data['visible'],
                "count": data["count"] if data['visible'] else -1} 
                for food_key, data in visible_foods.items()
            },
            "movable_spaces": movable_spaces
        }


    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]