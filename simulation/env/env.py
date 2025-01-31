from typing import Tuple
from copy import copy
from enum import Enum
import random
import math

import numpy as np

from pettingzoo import AECEnv
from gymnasium import spaces



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
    alive: bool
    
    def __init__(
        self,
        *,
        agent_id: str = "agent_id",
        hunger: float = 1,
        inventory: int = 0,
        reputation: float = 0,
        alive: bool = True
    ):
        self.agent_id = agent_id
        self.hunger = hunger
        self.inventory = inventory
        self.reputation = reputation
        self.alive = alive
    

class Villager(Agent):
    ...


class Thief(Agent):
    ...


class AgentActions(Enum):
    SKIP = 0
    MOVE = 1
    GATHER = 2
    TRADE = 3
    EAT = 4
    

class VillagerActions(AgentActions):
    EXECUTE = 5


class ThiefActions(AgentActions):
    STEAL = 6


class MovementDirection(Enum):
    SKIP = -1
    Forward = 0
    Backward = 1
    Leftward = 2
    Rightward = 3


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
    
    
    def __get_villager(self, villager_name: str) -> Villager | None:
        x, y = self.__agents_cahce[villager_name]
        return self.__grid[x][y].villagers.get(villager_name, None)
    
    
    def __get_thief(self, thief_name: str) -> Thief | None:
        x, y = self.__agents_cahce[thief_name]
        return self.__grid[x][y].thieves.get(thief_name, None)
    
    
    def __get_agent(self, agent_name: str) -> Agent | None:
        agent = self.__get_villager(agent_name)
        if agent is None:
            agent = self.__get_thief(agent_name)
        return agent
    
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
            "action_type": spaces.Discrete(len(VillagerActions)),
            "move_amount": spaces.Discrete(self.__max_agent_move_limit),
            "move_directions": spaces.Box(
                low=MovementDirection.SKIP, 
                high=MovementDirection.Rightward, 
                shape=(self.__max_agent_move_limit,),
                dtype=np.int32
            ),
            "gather_amount": spaces.Discrete(self.__max_agent_gather_limit),
            "trade_item": spaces.Box(
                low=[0, 0], 
                high=[self.__max_agent_inventory_limit, self.__grid_agents_count], 
                shape=(2,),
                dtype=np.int32
            ),
            "eat_item": spaces.Discrete(self.__max_agent_inventory_limit),
            "execute_target": spaces.Discrete(self.__grid_agents_count),
        })
    
    
    def __generate_thief_action_space(self) -> spaces.Dict:
        return spaces.Dict({
            "action_type": spaces.Discrete(len(ThiefActions)),
            "move_amount": spaces.Discrete(self.__max_agent_move_limit),
            "move_directions": spaces.Box(
                low=MovementDirection.SKIP, 
                high=MovementDirection.Rightward, 
                shape=(self.__max_agent_move_limit,),
                dtype=np.int32
            ),
            "gather_amount": spaces.Discrete(self.__max_agent_gather_limit),
            "trade_item": spaces.Box(
                low=[0, 0], 
                high=[self.__max_agent_inventory_limit, self.__grid_agents_count], 
                shape=(2,),
                dtype=np.int32
            ),
            "eat_item": spaces.Discrete(self.__max_agent_inventory_limit),
            "steal_target": spaces.Box(
                low=[0, 0], 
                high=[self.__max_agent_inventory_limit, self.__grid_agents_count], 
                shape=(2,),
                dtype=np.int32
            ),
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
                    "alive": spaces.Discrete(2),
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
                "alive": self.__grid[x][y].agents[other_name].alive
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
        
    
    def __agent_actions(self, agent: Agent, action_type: AgentActions, action: dict):
        x, y = self.__agents_cahce[agent.agent_id]
        
        match action_type:
            case AgentActions.SKIP:
                ...
            case AgentActions.GATHER:
                gather_amount: int = action["gather_amount"]
                            
                food_count = self.__grid[x][y].food_count 
                
                real_amount = max(0, food_count - gather_amount)
                inventory_limit = self.__max_agent_inventory_limit - agent.inventory
                
                real_amount = inventory_limit - real_amount
                
                agent.inventory += real_amount
                self.__grid[x][y].food_count -= real_amount
                
            case AgentActions.TRADE:
                trade_item: list[int] = action["trade_item"]
                
                food_count: int = trade_item[0]
                food_count = min(food_count, agent.inventory)
                
                other_agent_index: int = trade_item[1]
                other_agent_name = self.agents[other_agent_index]
                other_x, other_y = self.__agents_cahce[other_agent_name]
                
                other_agent = self.__get_agent(other_agent_name)
                if other_x != x or other_y != y or not other_agent.alive:
                    return
                
                food_count = max(self.__max_agent_inventory_limit - other_agent.inventory, food_count)
                
                other_agent.inventory += food_count
                agent.inventory -= food_count
                agent.reputation = min(1, agent.reputation + food_count * 0.1)
                
                
            case AgentActions.EAT:
                eat_item: int = action["eat_item"]
                
                eat_item = min(agent.inventory, eat_item)
                
                agent.inventory -= eat_item
                agent.hunger += min(1, agent.hunger + 0.1 * eat_item)
    
    def __get_after_movement_position(self, move_amount: int, move_directions: list[int], init_x: int, init_y: int):
        curr_x, curr_y = init_x, init_y
        for i in range(move_amount + 1):
            direction = MovementDirection(move_directions[i])
            match direction:
                case MovementDirection.SKIP:
                    ...
                case MovementDirection.Forward:
                    if self.is_cell_passable(curr_x, curr_y + 1):
                        curr_y += 1
                case MovementDirection.Backward:
                    if self.is_cell_passable(curr_x, curr_y - 1):
                        curr_y -= 1
                case MovementDirection.Rightward:
                    if self.is_cell_passable(curr_x + 1, curr_y):
                        curr_x += 1
                case MovementDirection.Leftward:
                    if self.is_cell_passable(curr_x - 1, curr_y):
                        curr_x -= 1
        
        return curr_x, curr_y
    
    def __villager_actions(self, villager: Villager, action_type: VillagerActions, action: dict):
        x, y = self.__agents_cahce[villager.agent_id]
        
        match action_type:
            case VillagerActions.EXECUTE:
                execute_target: int = action["execute_target"]
                
                other_agent_name = self.agents[execute_target]
                other_x, other_y = self.__agents_cahce[other_agent_name]
                
                other_agent = self.__get_agent(other_agent_name)
                if other_x != x or other_y != y or not other_agent.alive:
                    return
                
                other_agent.reputation = max(-1, other_agent.reputation - 0.5)
            case VillagerActions.MOVE:
                move_amount: int = action['move_amount']
                move_directions: list[int] = action['move_directions']
                
                init_x, init_y = self.__agents_cahce[villager.agent_id]
                curr_x, curr_y = self.__get_after_movement_position(move_amount, move_directions, init_x, init_y)
                
                del self.__grid[init_x][init_y].villagers[villager.agent_id]
                
                self.__grid[curr_x][curr_y].villagers[villager.agent_id] = villager
            case _:
                self.__agent_actions(villager, action_typ, action)
    
    
    def __theif_actions(self, thief: Thief, action_type: ThiefActions, action: dict):
        x, y = self.__agents_cahce[thief.agent_id]
        
        match action_type:
            case ThiefActions.STEAL:
                steal_target: list[int] = action["steal_target"]
                
                food_count: int = steal_target[0]
                other_agent_index: int = steal_target[1]
                
                other_agent_name = self.agents[other_agent_index]
                other_x, other_y = self.__agents_cahce[other_agent_name]
                
                other_agent = self.__get_agent(other_agent_name)
                if other_x != x or other_y != y or not other_agent.alive:
                    return
                
                food_count = min(food_count, self.__max_agent_inventory_limit - thief.inventory)
                food_count = min(food_count, other_agent.inventory)
                
                other_agent.inventory -= food_count
                thief.inventory += food_count
                
                dice = random.uniform(0, 1)
                if dice < food_count * 0.2:
                    thief.reputation = max(-1, thief.reputation - food_count * 0.1)
                
            case ThiefActions.MOVE:
                move_amount: int = action['move_amount']
                move_directions: list[int] = action['move_directions']
                
                init_x, init_y = self.__agents_cahce[thief.agent_id]
                curr_x, curr_y = self.__get_after_movement_position(move_amount, move_directions, init_x, init_y)
                
                del self.__grid[init_x][init_y].thieves[thief.agent_id]
                
                self.__grid[curr_x][curr_y].thieves[thief.agent_id] = thief
            case _:
                self.__agent_actions(thief, action_type, action)


    def step(self, actions):
        agent_name = self.agent_selection
        x, y = self.__agents_cahce[agent_name]
        villager: Villager | None = self.__grid[x][y].villagers.get(agent_name, None)
        thief: Thief | None = self.__grid[x][y].thieves.get(agent_name, None)
        agent_action = actions[agent_name]
        
        if villager is not None:
            villager_actions = VillagerActions(agent_action["action_type"])
            self.__villager_actions(villager, villager_actions)
        else:
            thief_actions = ThiefActions(agent_action["action_type"])
            self.__theif_actions(thief, thief_actions)
        

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]