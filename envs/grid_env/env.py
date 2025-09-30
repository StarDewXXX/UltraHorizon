import random
import asyncio
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import os
import json
import openai
from openai import OpenAI
from envs.common import Difficulty

ENV_PROMPT = """
=== GRID EXPLORATION GAME ===

GAME OBJECTIVE:
You are an agent in a 10x10 grid world. Your goal is to discover the hidden rules of effects of different letters (A-E).

BASIC RULES:
- You start at initial position with 20 energy points
- Each move costs 1 energy point
- You have a maximum of 30 steps per game round, and a required total steps of {required_steps}, you cannot move after reaching the limit, and you cannot commit before reaching the limit
- The grid contains letters A through E, each with unique hidden effects on your score or energy. X means you have already stepped on this tile and it has no effect.
- The effects of each letter is consistent across games, but you need to discover them through exploration.
- After you step on a tile, its letter changes to 'X'
- The game ends when you run out of energy or reach 30 steps

IMPORTANT NOTES:
- **Letter effects are HIDDEN but follow CONSISTENT rules** across different games and you must discover the rules through experimentation. The effects may be related to your position, energy, step count or visit count. 
- Each letter (A-E) has a different scoring mechanics that you need to figure out. The effect of each tile should not be concluded from a single try; multiple validations under different situations are essential to confirm that the observed outcome is not merely a random occurrence. You should **fully utilize the reset opportunity** to explore the effects of each letter.
- **Use notes to record your plans, ideas, and discoveries, and review them frequently to stay on track**. Also, record the effects into notes of each letter as you discover them. You may also need to record the changes after each move for later analysis.
- You can reset and replay multiple times to learn the patterns. Your goal is **not** to maximize the score, but to explore the mechanism clearly. It is recommended to make full use of each game until all steps or energy are exhausted, **avoiding resets too easily**. At the same time, **repeated trials should be conducted** to ensure the conclusions are correct before committing. **You can only commit your answer once and then the game ends**, commit only when you are confident about the effects of each letter.
- Since each step consumes 1 energy, it is therefore not counted as an effect.

COMMIT REQUIREMENTS:
- Complete coverage: You must provide exactly one effect description for each of A, B, C, D, and E.
- Trigger condition required: Your description must specify the underlying trigger condition, which may depend on: step count, visit count, energy, grid position. Stating only surface patterns such as “sometimes +2, sometimes -1” is not sufficient.
- No vague descriptions: Do not use terms like “random,” “uncertain,” or “complex.” Do not provide multiple conflicting explanations.
- Evaluation is strict: only if you clearly identify the correct trigger condition and effect will your answer be marked as correct.

COORDINATE FORMAT:
All positions are shown as (x,y,letter) where x=column, y=row, letter=current tile letter

Now start exploring and discovering the hidden mechanics.
"""

@dataclass
class GameState:
    x: int = field(default_factory=lambda: random.randint(0, 9))
    y: int = field(default_factory=lambda: random.randint(0, 9))
    energy: int = 20  # Initial energy m=20
    score: int = 0
    steps: int = 0
    max_steps: int = 30  # Maximum n=30 steps
    visited_tiles: Dict[Tuple[int, int], int] = field(default_factory=dict)  # Track visit counts
    visited_letters: Dict[str, int] = field(default_factory=dict)
    game_over: bool = False

    def add_score(self, points: int):
        self.score += points

    def change_energy(self, amount: int):
        self.energy += amount

    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < 10 and 0 <= y < 10

    def move_to(self, x: int, y: int) -> bool:
        if not self.is_valid_position(x, y) or self.energy <= 0 or self.game_over:
            return False
        
        self.x = x
        self.y = y
        self.energy -= 1
        self.steps += 1
        
        # Track visits to this tile
        pos = (x, y)
        self.visited_tiles[pos] = self.visited_tiles.get(pos, 0) + 1
        
        if self.energy <= 0 or self.steps >= self.max_steps:
            self.game_over = True
            
        return True

class MysteryGridEnvironment:
    def __init__(self, difficulty: Difficulty = Difficulty.HARD, required_steps: int = 50, free=False):
        self.max_resets = 20
        self.reset_count = 0
        self.difficulty = difficulty
        print(f"Environment initialized with difficulty: {self.difficulty}")
        self.state = GameState()
        self.grid = self._generate_grid()
        self.total_steps = 0
        self.required_steps = required_steps
        self.free = free
        self.committed = False
        self.final_result = {}
        self.judge_config = self.load_judge_config()
        if free:
            self.env_prompt = ENV_PROMPT.replace(", and a required total steps of {required_steps},",".")
        else:
            self.env_prompt = ENV_PROMPT.format(required_steps=self.required_steps)
        print("ENV PROMPT:\n", self.env_prompt)
        
        # Define all possible effects
        self.ALL_EFFECTS = {
            "effect_1": self._effect_1, 
            "effect_2": self._effect_2, 
            "effect_3": self._effect_3,
            "effect_4": self._effect_4,
            "effect_5": self._effect_5,
            "effect_6": self._effect_6,
            "effect_7": self._effect_7,
            "effect_8": self._effect_8,
            "effect_9": self._effect_9,
            "effect_10": self._effect_10,
        }
        
        # Define difficulty-specific effect sets
        self.DIFFICULTY_EFFECTS = {
            Difficulty.EASY: {
                "A": "effect_1",  
                "B": "effect_2",    
                "C": "effect_3",  
                "D": "effect_4",  
                "E": "effect_5",  
            },
            Difficulty.MEDIUM: {
                "A": "effect_3",  
                "B": "effect_5",  
                "C": "effect_6",  
                "D": "effect_7",  
                "E": "effect_8",  
            },
            Difficulty.HARD: {
                "A": "effect_6",  
                "B": "effect_7",  
                "C": "effect_8", 
                "D": "effect_9",  
                "E": "effect_10", 
            }
        }

    def _generate_grid(self) -> List[List[str]]:
        """Generate a 10x10 grid with letters A-E based on difficulty"""
        letters = ['A', 'B', 'C', 'D', 'E']
        grid = []
        for i in range(10):
            row = []
            for j in range(10):
                row.append(random.choice(letters))
            grid.append(row)
        return grid

    def _refresh_tile(self, x: int, y: int):
        """Refresh the letter at a visited position to X"""
        self.grid[9 - y][x] = 'X'

    # All possible effect implementations
    
    def _effect_1(self):
        """Simple +1 score"""
        self.state.add_score(1)

    def _effect_2(self):
        """Simple +1 energy"""
        self.state.change_energy(1)

    def _effect_3(self):
        """Position-based: odd coordinates +1, even coordinates -1"""
        if (self.state.x + self.state.y) % 2 == 1:
            self.state.add_score(1)
        else:
            self.state.add_score(-1)

    def _effect_4(self):
        """Simple -1 score"""
        self.state.add_score(-1)

    def _effect_5(self):
        """Energy threshold effect"""
        if self.state.energy >= 15:
            self.state.add_score(2)
        elif self.state.energy < 15:
            self.state.add_score(-2)

    def _effect_6(self):
        """Step count effect"""
        if self.state.steps % 3 == 0:
            self.state.add_score(2)
        else:
            self.state.add_score(-1)

    def _effect_7(self):
        """Corner and edge effects"""
        x, y = self.state.x, self.state.y
        if (x == 0 or x == 9) and (y == 0 or y == 9):  # Corners
            self.state.add_score(3)
        if x <= 1 or x >= 8 or y <= 1 or y >= 8: # Edges
            self.state.add_score(3)


    def _effect_8(self):
        """Visit count based effect - only for current tile"""
        letter = self.grid[9 - self.state.y][self.state.x]
        visit_count = self.state.visited_letters.get(letter, 0)
        
        if visit_count % 2 == 1:
            self.state.add_score(1)
        else:
            self.state.add_score(0)

    def _effect_9(self):
        """Energy management - penalty for low energy, bonus for high"""
        if self.state.energy < 10:
            self.state.add_score(-2)
            self.state.change_energy(10)
        elif self.state.energy >= 10:
            self.state.add_score(1)

    def _effect_10(self):
        """Complex position calculation"""
        score_change = self.state.x - self.state.y 
        self.state.add_score(score_change)

    def _effect_X(self):
        """X: No effect (visited tiles)"""
        pass

    async def move(self, direction: str) -> Dict[str, Any]:
        """[agent tool] Move the agent in specified direction (up/down/left/right). You can move for multiple times in each step."""
        # print("DEBUG", self.difficulty, self.DIFFICULTY_EFFECTS[self.difficulty])

        if self.state.game_over:
            return {"success": False, "message": "Game is over. Use reset to start a new game."}
        
        if self.total_steps >= self.required_steps:
            return {"success": False, "message": "Maximum total steps reached. Cannot move further. You should commit your answers."}
        
        if self.committed:
            return {"success": False, "message": "You have already committed your answers. No further moves allowed."}

        direction_map = {
            "up": (0, 1),
            "down": (0, -1),
            "left": (-1, 0),
            "right": (1, 0)
        }

        if direction.lower() not in direction_map:
            return {"success": False, "message": "Invalid direction. Use: up, down, left, right"}

        dx, dy = direction_map[direction.lower()]
        new_x, new_y = self.state.x + dx, self.state.y + dy

        if not self.state.move_to(new_x, new_y):
            return {"success": False, "message": "Invalid move or out of energy/steps"}
        else:
            self.total_steps += 1

        letter = self.grid[9 - new_y][new_x]

        # Update visit count for this letter type
        if letter != 'X':
            self.state.visited_letters[letter] = self.state.visited_letters.get(letter, 0) + 1

        if letter == 'X':
            pass
        elif letter in self.DIFFICULTY_EFFECTS[self.difficulty]:
            effect_name = self.DIFFICULTY_EFFECTS[self.difficulty][letter]
            self.ALL_EFFECTS[effect_name]()

        self._refresh_tile(new_x, new_y)

        return {
            "success": True,
            "position": f"({new_x},{new_y},{letter})",
            "energy": self.state.energy,
            "score": self.state.score,
            "steps": self.state.steps,
            "game_over": self.state.game_over,
            "difficulty": self.difficulty.value,
            "remain_reset_times": self.max_resets - self.reset_count
        }

    async def get_current_state(self) -> Dict[str, Any]:
        """[agent tool] Get current game state and nearby tiles"""
        nearby_tiles = []
        for dx in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                x, y = self.state.x + dx, self.state.y + dy
                if self.state.is_valid_position(x, y):
                    nearby_tiles.append(f"({x},{y},{self.grid[9 - y][x]})")

        return {
            "current_position": f"({self.state.x},{self.state.y},{self.grid[9 - self.state.y][self.state.x]})",
            "energy": self.state.energy,
            "score": self.state.score,
            "steps": self.state.steps,
            "max_steps_in_this_round": self.state.max_steps,
            "nearby_tiles": nearby_tiles,
            "game_over": self.state.game_over,
            "difficulty": self.difficulty.value
        }

    async def get_full_map(self) -> Dict[str, Any]:
        """[agent tool] Get the complete map state with coordinates"""
        map_data = []
        for y in range(10):
            for x in range(10):
                # Convert mathematical coordinates to display coordinates
                map_data.append(f"({x},{9-y},{self.grid[y][x]})")
        
        return {
            "map": map_data,
            "agent_position": f"({self.state.x},{self.state.y})",
            "difficulty": self.difficulty.value
        }

    async def reset(self) -> Dict[str, Any]:
        """[agent tool] Reset the environment for a new game, optionally with new difficulty"""
        
        self.reset_count += 1
        self.state = GameState()
        self.grid = self._generate_grid()
        
        return {
            "success": True,
            "message": f"Environment reset. Reset count: {self.reset_count}",
            "initial_position": f"({self.state.x},{self.state.y},{self.grid[9 - self.state.y][self.state.x]})",
            "energy": self.state.energy,
            "max_steps_in_this_round": self.state.max_steps,
            "difficulty": self.difficulty.value
        }
    
    def load_judge_config(self):
        """Load judge model configuration from a YAML file"""
        import yaml
        config_path = 'judge_config.yaml' #os.path.join(os.path.dirname(__file__), 'judge_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Judge config loaded:", config)
        return config

    async def commit_final_result(self, content: str) -> Dict[str, Any]:
        """
        [agent tool] Submit the complete and precise mapping between letters (A-E) and their corresponding effects for final evaluation. **You can only commit once**. **Before commit, yo must check your notes and analyze them.** Only commit after you have fully explored the grid, conducted sufficient trials, and are confident that you understand the exact effect rules of each letter under all situations. **Submitting incomplete, uncertain, or partially inferred effects will be considered incorrect**.
        """

        # cannot commit if not reach minimum interaction steps if still can reset

        if not self.free:
            if self.total_steps < self.required_steps and self.reset_count < self.max_resets:
                return {
                    "success": False,
                    "message": f"Cannot commit yet. Total move steps required: {self.required_steps}, current steps: {self.total_steps}. You should do more exploration and analysis to validate your answers before committing."
                }

        # Create effect descriptions mapping
        effect_descriptions = {
            "effect_1": "Simple +1 score",
            "effect_2": "Simple +1 energy", 
            "effect_3": "Position-based: odd coordinates (x+y) +1 score, even coordinates (x+y) -1 score",
            "effect_4": "Simple -1 score",
            "effect_5": "Energy threshold effect: if energy >= 15 then +2 score, if energy < 15 then -2 score",
            "effect_6": "Step count effect: if steps % 3 == 0 then +2 score, else -1 score",
            "effect_7": "Corner and edge effects: corners give +3 score, edges give +3 score, otherwise no effect. Formally, if x in (0, 1, 8, 9) or y in (0, 1, 8, 9): score+=3",
            "effect_8": "Visit count based effect: +1 score if this letter type has been visited odd number of times, 0 score if even number of times",
            "effect_9": "Energy management: if energy is low then -2 score and +10 energy, if energy is high then +1 score",
            "effect_10": "Complex position calculation: score change = x - y"
        }
        
        # Get the effect name mapping for current difficulty
        effect_name_mapping = self.DIFFICULTY_EFFECTS[self.difficulty]
        
        # Create ground truth with actual effect descriptions
        ground_truth = {}
        for letter, effect_name in effect_name_mapping.items():
            ground_truth[letter] = effect_descriptions[effect_name]
        
        client = OpenAI(
            api_key=self.judge_config.get("api_key"),
            base_url=self.judge_config.get("base_url"),
        )
        prompt = """
You are an evaluator. Compare the agent's submitted mapping of letters to effect descriptions against the ground truth mapping.
The agent should have inferred what each letter (A, B, C, D, E) does when stepped on.

<Ground truth mapping>:
{ground_truth}

<Agent submitted mapping>:
{content}

Compare the agent's descriptions with the ground truth effects. A mapping is considered correct if the agent's description captures the essential rule of the effect, even if the wording is different. Since each step consumes 1 energy, it is therefore not counted as an effect. However, it is also correct if the agent specifies a consumption of 1 energy for each effect. You should follow these rules to judge:
Evaluate correctness according to these strict rules:

1. **Trigger Condition Must Be Identified**
An effect description is correct **only if** the agent identifies the underlying *trigger condition* (e.g., step count, visit count parity, energy level, or coordinate-based calculation).
Simply describing observed outcome patterns (e.g., “sometimes +2, sometimes -1”, “alternating pattern”, “mostly +1 with exceptions”) **without stating the actual trigger condition** is incorrect.

2. **Equivalent Wording Allowed**
Different wording is acceptable only if it expresses the *same logical condition*.
Example: “odd/even visits” ≡ “visit count parity”.
Example: “boundary positions” ≡ “corners and edges”.
Descriptions that skip the explicit condition and only give surface observations are incorrect.

3. **Vague or Wrong Attributions are Incorrect**
If the agent uses vague terms such as “variable”, “random”, “uncertain”, or “complex behavior” instead of specifying the true condition, it is incorrect.
If the agent attributes the effect to the wrong factor (e.g., says “position-dependent” when the actual rule depends on step count or energy), it is incorrect.

4. **One-to-One Mapping Requirement**
Each symbol must map to exactly one effect rule.
Multiple conflicting explanations, incomplete conditions, or missing mappings are incorrect.

5. **Evaluation Criteria**
Mark **Correct** only if:
The trigger condition identified by the agent matches the ground truth condition.
The outcome description aligns with the ground truth rule.
Otherwise, mark **Incorrect**.

Format your answer as json:

{
  "final_score": 40,
  "score_breakdown": [
    {"criterion": "A", "max_score": 20, "awarded_score": 20, "comment": <your comment>},
    {"criterion": "B", "max_score": 20, "awarded_score": 0, "comment": <your comment>},
    {"criterion": "C", "max_score": 20, "awarded_score": 0, "comment": <your comment>},
    {"criterion": "D", "max_score": 20, "awarded_score": 20, "comment": <your comment>},
    {"criterion": "E", "max_score": 20, "awarded_score": 0, "comment": <your comment>},
  ]
}
    """

        prompt = prompt.replace("{content}", content)
        prompt = prompt.replace("{ground_truth}", json.dumps(ground_truth, indent=2))

        print("[Evaluation Prompt]:\n", prompt)
        try:
            response = client.chat.completions.create(
                model=self.judge_config.get("model"),
                messages=[
                    {"role": "system", "content": "You are a precise evaluator of rules."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            # Get judge output text
            msg = response.choices[0].message
            judge_text = getattr(msg, "content", None)
            if "<think>" in judge_text and "</think>" in judge_text:
                judge_text = judge_text.split("</think>")[-1].strip()
            if judge_text is None and isinstance(msg, dict):
                judge_text = msg.get("content", "")
            judge_text = (judge_text or "").strip()

            # (Optional) Remove ```json fence
            if judge_text.startswith("```"):
                judge_text = judge_text.strip("`")
                # Simple processing to prevent prefix like json\n
                if judge_text.startswith("json"):
                    judge_text = judge_text[4:].lstrip()

            try:
                judge_result = json.loads(judge_text)
            except Exception:
                judge_result = {"raw_output": judge_text}

            output = {
                "judge_input": content,
                "judge_result": judge_result
            }
            self.final_result = output
            self.committed = True

            return {"success": True, "result": output}

        except Exception as e:
            return {
                "success": False,
                "message": f"Evaluation failed: {e}"
            }

    def get_difficulty_info(self) -> Dict[str, Any]:
        """Get information about current difficulty and its effects"""
        current_effects = self.DIFFICULTY_EFFECTS[self.difficulty]
        return {
            "difficulty": self.difficulty.value,
            "letter_effects": {letter: effect_name for letter, effect_name in current_effects.items()},
            "available_letters": list(current_effects.keys())
        }

async def terminal_game():
    env = MysteryGridEnvironment(difficulty=Difficulty.EASY)
    await env.reset()
    output = await env.commit_final_result("hahahahah")
    print("Commit Result:", output)

if __name__ == "__main__":
    asyncio.run(terminal_game())
