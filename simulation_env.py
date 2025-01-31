from typing import Tuple
from copy import copy
from enum import Enum
import random
import math

import numpy as np

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium import spaces
import pygame


class GridItem:
    pos_x: int
    pos_y: int

    def __init__(self, pos_x: int, pos_y: int) -> None:
        self.pos_x = pos_x
        self.pos_y = pos_y


class Obstacle(GridItem):
    ...


class Agent:
    index: int
    agent_id: str
    hunger: float # [0, 1]
    inventory: int
    reputation: float # [-1, 1]
    alive: bool
    
    def __init__(
        self,
        *,
        index: int = 0,
        agent_id: str = "agent_id",
        hunger: float = 1,
        inventory: int = 0,
        reputation: float = 0,
        alive: bool = True,
    ):
        self.index = index
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
    EXECUTE = 5
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
    ):
        self.food_count = food_count
        self.villagers = dict()
        self.thieves = dict()
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

    __agents_cache: dict[str, Tuple[int, int]]
    __food_cahce: dict[int, Tuple[int, int, int]]
    __obstacle_cache: dict[int, Tuple[int, int]]

    def __init__(
        self,
        *,
        length: int = 14, 
        food_generation_rate: float = 0.02,
        max_agent_sight_distance: int = 4,
        max_agent_inventory_limit: int = 3,
        max_agent_gather_limit: int = 2,
        max_agent_move_limit: int = 3,
        villager_count: int = 4,
        thief_count: int = 3,
        obstacle_count: int = 7,
        render_mode: str = None,
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
        
        self.__possible_villagers = [f"villager_{i}" for i in range(villager_count)]
        self.__possible_thieves = [f"thief_{i}" for i in range(thief_count)]
        
        self.possible_agents = self.__possible_villagers + self.__possible_thieves  
        self.render_mode = render_mode      
    
    
    def get_food_quantity(self) -> int:
        return self.__grid_food_count
    
    
    def is_cell_passable(self, x: int, y: int) -> bool:
        return (0 <= y < self.__length) and (0 <= x < self.__length) and not isinstance(self.__grid[x][y], Obstacle)
    
    
    def __get_villager(self, villager_name: str) -> Villager | None:
        x, y = self.__agents_cache[villager_name]
        return self.__grid[x][y].villagers.get(villager_name, None)
    
    
    def __get_thief(self, thief_name: str) -> Thief | None:
        x, y = self.__agents_cache[thief_name]
        return self.__grid[x][y].thieves.get(thief_name, None)
    
    
    def __get_agent(self, agent_name: str) -> Agent | None:
        agent = self.__get_villager(agent_name)
        if agent is None:
            agent = self.__get_thief(agent_name)
        return agent
    
    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        print(self._cumulative_rewards)
        self.__agents_cache = {}
        self.__food_cahce = {}
        self.__obstacle_cache = {}
        self.__generate_grid()
        self.turn = 0
        self.round = 0
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
    
    def __generate_obstacle(self, pos_x: int, pos_y: int):
        self.__obstacle_cache[pos_x * self.__length + pos_y] = (pos_x, pos_y)
        return Obstacle(pos_x=pos_x, pos_y=pos_y)
    
    
    def __generate_not_obstacle(self, pos_x: int, pos_y: int):
        return NotObstacle(pos_x=pos_x, pos_y=pos_y)
    
    
    def __generate_villager_action_space(self) -> spaces.Dict:
        return spaces.Dict({
            "action_type": spaces.Discrete(len(AgentActions) - 1),
            "move_amount": spaces.Discrete(self.__max_agent_move_limit),
            "move_directions": spaces.Box(
                low=MovementDirection.SKIP.value, 
                high=MovementDirection.Rightward.value, 
                shape=(self.__max_agent_move_limit,),
                dtype=np.int32
            ),
            "gather_amount": spaces.Discrete(self.__max_agent_gather_limit),
            "trade_amount": spaces.Discrete(self.__max_agent_inventory_limit),
            "eat_amount": spaces.Discrete(self.__max_agent_inventory_limit),
            "target_agent": spaces.Discrete(self.__grid_agents_count),
        })
    
    
    def __generate_thief_action_space(self) -> spaces.Dict:
        return spaces.Dict({
            "action_type": spaces.Discrete(len(AgentActions) - 1),
            "move_amount": spaces.Discrete(self.__max_agent_move_limit),
            "move_directions": spaces.Box(
                low=MovementDirection.SKIP.value, 
                high=MovementDirection.Rightward.value, 
                shape=(self.__max_agent_move_limit,),
                dtype=np.int32
            ),
            "gather_amount": spaces.Discrete(self.__max_agent_gather_limit),
            "trade_amount": spaces.Discrete(self.__max_agent_inventory_limit),
            "target_agent": spaces.Discrete(self.__grid_agents_count),
            "eat_amount": spaces.Discrete(self.__max_agent_inventory_limit),
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
        self.__agents_cache[villager_name] = (x, y)
        index = int(villager_name[-1])
        villager = Villager(agent_id=villager_name, index=index)
        action_space = self.__generate_villager_action_space()
        observation_space = self.__generate_agent_observation_space(villager_name)
        
        self.__grid[x][y].villagers[villager_name] = villager
        self.action_spaces[villager_name] = action_space
        self.observation_spaces[villager_name] = observation_space
    
    
    def __generate_thief(self, x: int, y: int, thief_name: str):
        self.__agents_cache[thief_name] = (x, y)
        index = int(thief_name[-1])
        thief = Thief(agent_id=thief_name, index=index)
        action_space = self.__generate_thief_action_space()
        observation_space = self.__generate_agent_observation_space(thief_name)
        
        self.__grid[x][y].thieves[thief_name] = thief
        self.action_spaces[thief_name] = action_space
        self.observation_spaces[thief_name] = observation_space
    
    
    def __generate_grid(self):
        self.__grid = [[self.__generate_not_obstacle(x, y) for y in range(self.__length)] for x in range(self.__length)]
        self.__availible_positions = {
            x * self.__length + y: (x, y) for y in range(self.__length) for x in range(self.__length)
        }
        self.action_spaces = dict()
        self.observation_spaces = dict()
        
        for _ in range(self.__grid_obstacle_count):
            rand_x, rand_y = random.choice(list(self.__availible_positions.values()))
            self.__grid[rand_x][rand_y] = self.__generate_obstacle(rand_x, rand_y)
            self.__availible_positions.pop(rand_x * self.__length + rand_y)
        
        
        for _ in range(self.__grid_food_count):
            rand_x, rand_y = random.choice(list(self.__availible_positions.values()))
            self.__generate_food(rand_x, rand_y)
        
        
        for villager_name in self.__possible_villagers:
            rand_x, rand_y = random.choice(list(self.__availible_positions.values()))
            self.__generate_villager(rand_x, rand_y, villager_name)
        
        
        for thief_name in self.__possible_thieves:
            rand_x, rand_y = random.choice(list(self.__availible_positions.values()))
            self.__generate_thief(rand_x, rand_y, thief_name)


    def __regenerate_food(self):
        for _ in range(self.__grid_food_count):
            rand_x, rand_y = random.choice(list(self.__availible_positions.values()))
            self.__generate_food(rand_x, rand_y)

    
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
            if not self.is_cell_passable(x, y) and not looking_for_an_obstacle:
                return False
        
        return True

    
    def observe(self, agent_name: str):
        x, y = self.__agents_cache[agent_name]
        agent: Agent = self.__get_agent(agent_name)
        
        visible_agents = {
            other_name: {
                "visible": self.__can_see(self.__grid[x][y], self.__grid[other_x][other_y]), 
                "position": np.ndarray([other_x, other_y])
            }
            for other_name, (other_x, other_y) in self.__agents_cache.items()
            if other_name != agent_name
        }
        
        reputations = {
            other_name: {
                "reputation": self.__get_agent(other_name).reputation,
                "alive": self.__get_agent(other_name).alive
            }
            for other_name in self.__agents_cache.keys()
            if other_name != agent_name
        }
        
        visible_foods = {
            food_key: {
                "visible": self.__can_see(self.__grid[x][y], self.__grid[food_x][food_y]), 
                "position": np.ndarray([food_x, food_y]),
                "count": food_count
            }
            for food_key, (food_x, food_y, food_count) in self.__food_cahce.items()
        }
        
        movable_spaces = np.full((self.__max_agent_move_limit * 2 + 1, self.__max_agent_move_limit * 2 + 1), -1)
        
        origin_x = x - self.__max_agent_move_limit
        origin_y = y - self.__max_agent_move_limit
        
        for i in range(movable_spaces.shape[0]):  # i goes 0..(2*limit)
            for j in range(movable_spaces.shape[1]):  # j goes 0..(2*limit)
                grid_x = origin_x + i
                grid_y = origin_y + j
                if 0 <= grid_x < self.__length and 0 <= grid_y < self.__length:
                    if self.is_cell_passable(grid_x, grid_y):
                        dist = math.dist((grid_x, grid_y), (x, y))
                        movable_spaces[i, j] = round(dist)
            
        return {
            "position": np.ndarray([x, y]),
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
        x, y = self.__agents_cache[agent.agent_id]
        
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
                
                self.__food_cahce[x * self.__length + y] = (x, y, self.__grid[x][y].food_count)
                
            case AgentActions.TRADE:
                food_count: int = action["trade_amount"]
                food_count = min(food_count, agent.inventory)
                
                other_agent_index: int = action["target_agent"]
                other_agent_name = self.agents[other_agent_index]
                other_x, other_y = self.__agents_cache[other_agent_name]
                
                other_agent = self.__get_agent(other_agent_name)
                if other_x != x or other_y != y or not other_agent.alive:
                    return
                
                food_count = max(self.__max_agent_inventory_limit - other_agent.inventory, food_count)
                
                other_agent.inventory += food_count
                agent.inventory -= food_count
                agent.reputation = min(1, agent.reputation + food_count * 0.1)
                
                
            case AgentActions.EAT:
                eat_amount: int = action["eat_amount"]
                
                eat_amount = min(agent.inventory, eat_amount)
                
                agent.inventory -= eat_amount
                agent.hunger += min(1, agent.hunger + 0.1 * eat_amount)
                
    
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
    
    def __villager_actions(self, villager: Villager, action_type: AgentActions, action: dict):
        if not villager.alive:
            return
        
        x, y = self.__agents_cache[villager.agent_id]
        
        match action_type:
            case AgentActions.EXECUTE:
                execute_target: int = action["target_agent"]
                
                other_agent_name = self.agents[execute_target]
                other_x, other_y = self.__agents_cache[other_agent_name]
                
                other_agent = self.__get_agent(other_agent_name)
                if other_x != x or other_y != y or not other_agent.alive:
                    return
                
                other_agent.reputation = max(-1, other_agent.reputation - 0.5)
            case AgentActions.MOVE:
                move_amount: int = action['move_amount']
                move_directions: list[int] = action['move_directions']
                
                init_x, init_y = self.__agents_cache[villager.agent_id]
                curr_x, curr_y = self.__get_after_movement_position(move_amount, move_directions, init_x, init_y)
                
                del self.__grid[init_x][init_y].villagers[villager.agent_id]
                
                self.__grid[curr_x][curr_y].villagers[villager.agent_id] = villager
                
                self.__agents_cache[villager.agent_id] = (curr_x, curr_y)
            case _:
                self.__agent_actions(villager, action_type, action)
    
    
    def __thief_actions(self, thief: Thief, action_type: AgentActions, action: dict):
        if not thief.alive:
            return

        x, y = self.__agents_cache[thief.agent_id]
        
        match action_type:
            case AgentActions.STEAL:                
                food_count: int = action["gather_amount"]
                other_agent_index: int = action["target_agent"]
                
                other_agent_name = self.agents[other_agent_index]
                other_x, other_y = self.__agents_cache[other_agent_name]
                
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
                
            case AgentActions.MOVE:
                move_amount: int = action['move_amount']
                move_directions: list[int] = action['move_directions']
                
                init_x, init_y = self.__agents_cache[thief.agent_id]
                curr_x, curr_y = self.__get_after_movement_position(move_amount, move_directions, init_x, init_y)
                
                del self.__grid[init_x][init_y].thieves[thief.agent_id]
                
                self.__grid[curr_x][curr_y].thieves[thief.agent_id] = thief
                self.__agents_cache[thief.agent_id] = (curr_x, curr_y)
            case _:
                self.__agent_actions(thief, action_type, action)


    def step(self, actions):
        agent_name = self.agent_selection
        
        if (
            self.terminations[agent_name]
            or self.truncations[agent_name]
        ):
            self._was_dead_step(actions)
            return
        
        agent = self.__get_agent(agent_name)
        
        if agent.hunger == -1 or agent.reputation == -1:
            agent.alive = False
            #self.terminations[agent_name] = True
            #self.truncations[agent_name] = True

        self.observations[agent_name] = self.observe(agent_name)
        self.state[agent_name] = self.observations[agent_name]
        
        if isinstance(agent, Villager):
            villager_action = AgentActions(actions["action_type"])
            self.__villager_actions(agent, villager_action, actions)
        else:
            thief_action = AgentActions(actions["action_type"])
            self.__thief_actions(agent, thief_action, actions)
        
        self.observations[agent_name] = self.observe(agent_name)
        self.state[agent_name] = self.observations[agent_name]
        self.round += 1
        
        if self._agent_selector.is_last():
            self.__regenerate_food()
            
            self.rewards = dict()
            
            self.turn += 1
            self.round = 0
            
            for agent_name in self.__agents_cache:
                agent = self.__get_agent(agent_name)
                self.rewards[agent_name] = 0.4 * (agent.inventory / self.__max_agent_inventory_limit) + \
                    0.2 * agent.reputation - 0.4 * (1 - agent.hunger)
        
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
    
    
    def __pygame_render(self, fps: int = 60) -> None:
        """
        Render a simple 2D view of the grid using pygame.
        Draw each agent index on the cell, stacked vertically if multiple.
        :param fps: Frames per second limit for rendering.
        """
        # Lazy init pygame once
        if not hasattr(self, '_render_initialized') or not self._render_initialized:
            pygame.init()
            
            # Tweakable parameters
            self._cell_size = 40
            self._window_width = self.__length * self._cell_size
            self._window_height = self.__length * self._cell_size
            
            # Create the window
            self._screen = pygame.display.set_mode((self._window_width, self._window_height))
            pygame.display.set_caption("Simulation Environment")
            
            # Font for drawing agent indices
            self._font = pygame.font.SysFont("Arial", 14)
            
            # For controlling the render frame rate
            self._clock = pygame.time.Clock()
            
            self._render_initialized = True

        # Process events (so the window remains responsive)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._render_initialized = False
                return

        # Fill background with white
        self._screen.fill((255, 255, 255))

        # Draw each cell in the grid
        for x in range(self.__length):
            for y in range(self.__length):
                cell = self.__grid[x][y]

                # Flip y-axis so that y=0 is at the bottom
                rect = pygame.Rect(
                    x * self._cell_size,
                    (self.__length - 1 - y) * self._cell_size,
                    self._cell_size,
                    self._cell_size
                )

                # Draw obstacle
                if isinstance(cell, Obstacle):
                    pygame.draw.rect(self._screen, (0, 0, 0), rect)
                    continue

                # Start with a light gray background
                pygame.draw.rect(self._screen, (220, 220, 220), rect)

                # Check what’s in the cell
                num_villagers = len(cell.villagers)
                num_thieves = len(cell.thieves)
                food_count = cell.food_count

                # If we have both villagers and thieves => color it purple
                if num_villagers > 0 and num_thieves > 0:
                    pygame.draw.rect(self._screen, (128, 0, 128), rect)
                # If only villagers => color green
                elif num_villagers > 0:
                    pygame.draw.rect(self._screen, (0, 200, 0), rect)
                # If only thieves => color red
                elif num_thieves > 0:
                    pygame.draw.rect(self._screen, (200, 0, 0), rect)

                # Draw a small brown circle for food (if any)
                if food_count > 0:
                    radius = 6
                    center_x = x * self._cell_size + radius + 4
                    center_y = (self.__length - 1 - y) * self._cell_size + self._cell_size - radius - 4
                    pygame.draw.circle(self._screen, (139, 69, 19), (center_x, center_y), radius)

                # -- Draw agent labels in the cell (stacked) --
                # We'll place them near the top-left corner of each cell
                # Stacking them vertically with small offsets.
                label_x = rect.left + 4
                label_y = rect.top + 4  # We'll increment this for each agent

                # Create a list of label strings for each agent
                agent_labels = []
                
                # Villagers
                for villager_name, villager_obj in cell.villagers.items():
                    # E.g. "villager_3" => get last chunk after underscore => "3"
                    # or simply villager_name[-1] if single-digit
                    agent_index = villager_name.split('_')[-1]
                    label_str = f"V{agent_index}"
                    agent_labels.append(label_str)

                # Thieves
                for thief_name, thief_obj in cell.thieves.items():
                    agent_index = thief_name.split('_')[-1]
                    label_str = f"T{agent_index}"
                    agent_labels.append(label_str)

                # Now draw them
                for label_str in agent_labels:
                    # Render the text
                    text_surface = self._font.render(label_str, True, (255, 255, 255))  # White text
                    self._screen.blit(text_surface, (label_x, label_y))
                    label_y += 15  # move down so next label doesn't overlap

        # Update the display
        pygame.display.flip()
        self._clock.tick(fps)


    def render(self):
        self.__pygame_render()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = SimulationEnviornment(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env