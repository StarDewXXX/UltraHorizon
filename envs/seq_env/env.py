import asyncio
import random
from enum import Enum
from typing import List, Dict, Tuple, Optional
from typing import Dict, List, Tuple, Any, Callable
import json
import openai
from openai import OpenAI
from envs.common import Difficulty

ENV_PROMPT = '''
=== SEQUENCE EXPLORATION GAME ===

GAME OBJECTIVE:
You are an agent facing a hidden dual-sequence transformation system. Your goal is to discover the hidden rules applied to pairs of sequences of letters (A-E).

BASIC RULES:
- You must input TWO sequences of EXACTLY 5 CHARACTERS each, using only letters A-E
- Each sequence must contain at least 2 different letters from A-E
- The first sequence is the "main str" and the second is the "vice str"
- Each input pair consumes 1 step. There is a required total steps of {required_steps} to complete the game
- After inputting a sequence pair, they will undergo a fixed chain of transformations
- All transformations apply the same rules consistently, operating on current results and/or original inputs
- The same hidden rules apply consistently across the game, but you must discover them through exploration
- Game ends when you reach the maximum step count or submit your final answer

IMPORTANT NOTES:
- Effects are deterministic but hidden; repeated testing is needed to confirm true rules
- Each rule may depend on sequence content, position, history, or relationships between main/vice
- Use notes to record observed transformations and hypotheses
- Do not commit your final answer until you are confident, as you can only commit once
- The goal is to uncover and describe the mechanisms behind dual-sequence transformations

SEQUENCE FORMAT:
Inputs are pairs like ("ABCDE", "DECBA"), ("AABCE", "DDEAC"). Each must be exactly 5 characters with at least 2 different letters.
Outputs are shown step by step with rule applications.

DIFFICULTY LEVELS:
- Easy: Uses simplified versions of rules 1-5
- Hard: Uses complex versions of rules 1-5

Now begin experimenting and uncover the transformation rules.
'''

def is_prime(n: int) -> bool:
    """Check if a number is prime"""
    if n <= 1:  # Numbers less than or equal to 1 are not prime
        return False
    if n == 2:  # 2 is prime
        return True
    if n % 2 == 0:  # Even numbers (except 2) are not prime
        return False

    # Only need to check up to sqrt(n)
    limit = int(n ** 0.5) + 1
    for i in range(3, limit, 2):  # Only check odd numbers
        if n % i == 0:
            return False
    return True

class SequenceExploreEnvironment:
    def __init__(self, difficulty: Difficulty = Difficulty.EASY, required_steps: int = 10, free=False):
        self.difficulty = difficulty
        self.total_steps = 0
        self.alphabet = ['A', 'B', 'C', 'D', 'E']
        self.history = []
        self.final_result = {}
        self.committed = False
        
        # Initialize all possible rules
        self.ALL_RULES = {
            "rule_1": self._rule_1, 
            "rule_2": self._rule_2, 
            "rule_3": self._rule_3,
            "rule_4": self._rule_4,
            "rule_5": self._rule_5,
            "rule_6": self._rule_6,
            "rule_7": self._rule_7,
            "rule_8": self._rule_8,
            "rule_9": self._rule_9,
            "rule_10": self._rule_10,
        }
        
        # Initialize after setting up the mappings
        self.required_steps = required_steps
        self.free = free
        self.current_rules = self._initialize_rules()
        
        # Add judge config for evaluation
        self.judge_config = self.load_judge_config()
        # self.env_prompt = ENV_PROMPT.format(required_steps=self.required_steps)
        if free:
            self.env_prompt = ENV_PROMPT.replace(" There is a required total steps of {required_steps} to complete the game.","")
        else:
            self.env_prompt = ENV_PROMPT.format(required_steps=self.required_steps)
        
        print("ENV PROMPT:\n", self.env_prompt)
    
    def load_judge_config(self):
        """Load judge model configuration from a YAML file"""
        import yaml
        config_path = 'judge_config.yaml' #os.path.join(os.path.dirname(__file__), 'judge_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Judge config loaded:", config)
        return config
    
    def _initialize_rules(self) -> List[str]:
        if self.difficulty == Difficulty.EASY:
            return ["rule_1", "rule_2", "rule_3", "rule_4", "rule_5"]
        else:  # HARD
            return ["rule_6", "rule_7", "rule_8", "rule_9", "rule_10"]
    
    def _get_external_rule_name(self, internal_rule_name: str) -> str:
        """Convert internal rule names to external display names (always rule_1 to rule_5)"""
        rule_mapping = {
            "rule_1": "rule_1", "rule_2": "rule_2", "rule_3": "rule_3", 
            "rule_4": "rule_4", "rule_5": "rule_5",
            "rule_6": "rule_1", "rule_7": "rule_2", "rule_8": "rule_3",
            "rule_9": "rule_4", "rule_10": "rule_5"
        }
        return rule_mapping.get(internal_rule_name, internal_rule_name)
    
    def _get_last_main(self) -> str:
        """Get the main string from the last input"""
        if len(self.history) > 0:
            return self.history[-1]["main_input"]
        return ""
    
    # Rule implementations
    def _rule_1(self, current: str, main: str, vice: str) -> str:
        """Easy: Alternate characters from main and vice (simplified interleaving)"""
        if not main or not vice:
            return current
        result = ""
        for i in range(min(len(main), len(vice))):
            result += main[i] + vice[i]
        return result[:10]  # Limit length
    
    def _rule_2(self, current: str, main: str, vice: str) -> str:
        """Easy: Simple addition of character positions"""
        if not main or not vice:
            return current.replace('A', 'X')  # Fallback
        result = ""
        for i in range(min(len(main), len(vice))):
            main_val = ord(main[i]) - ord('A')
            vice_val = ord(vice[i]) - ord('A')
            combined_val = (main_val + vice_val)
            result += chr(ord('A') + combined_val)
        return result
    
    def _rule_3(self, current: str, main: str, vice: str) -> str:
        """Easy: Take maximum character at each position"""
        if not main or not vice:
            # Remove consecutive duplicates
            if not current:
                return current
            result = current[0]
            for i in range(1, len(current)):
                if current[i] != current[i-1]:
                    result += current[i]
            return result
        
        result = ""
        for i in range(min(len(main), len(vice))):
            result += max(main[i], vice[i])
        return result
    
    def _rule_4(self, current: str, main: str, vice: str) -> str:
        """Easy: Simple comparison and length append"""
        if not main or not vice:
            # Append length
            length = len(current)
            if length <= 26:
                length_char = chr(ord('A') + length - 1)
                return current + length_char
            return current
        
        result = ""
        for i in range(min(len(main), len(vice))):
            if main[i] >= vice[i]:
                result += main[i]
            else:
                result += vice[i]
        return result
    
    def _rule_5(self, current: str, main: str, vice: str) -> str:
        """Easy: Simple sorting"""
        if not main or not vice:
            return ''.join(sorted(current.upper()))
        
        # Combine and sort
        combined = main + vice
        return ''.join(sorted(combined))
    
    def _rule_6(self, current: str, main: str, vice: str) -> str:
        """Hard: Interweave main and vice characters, order depends on total_steps"""
        if not main or not vice:
            return current
        
        result = []
        max_len = max(len(main), len(vice))
        
        main_first = (self.total_steps % 2 == 1)  # Odd -> main first, even -> vice first
        
        for i in range(max_len):
            if main_first:
                if i < len(main):
                    result.append(main[i])
                if i < len(vice):
                    result.append(vice[i])
            else:
                if i < len(vice):
                    result.append(vice[i])
                if i < len(main):
                    result.append(main[i])
        
        return "".join(result)
    
    def _rule_7(self, current: str, main: str, vice: str) -> str:
        """Hard: Reverse and concatenate with +1 shift"""
        if not current:
            return current
        
        # Reverse current
        str_r = current[::-1]
        
        # Shift all characters by +1 (A->B, B->C, ..., E->A)
        def shift_char(c):
            if c in self.alphabet:
                idx = (ord(c) - ord('A') + self.total_steps) % 26
                return chr(ord('A') + idx)
            return c
        
        str_r_shifted = ''.join(shift_char(c) for c in str_r)
        current_shifted = ''.join(shift_char(c) for c in current)
        
        return str_r_shifted + current_shifted
    
    def _rule_8(self, current: str, main: str, vice: str) -> str:
        """Hard: Copy character based on step number"""
        if not current:
            return current
        
        n = self.total_steps
        a = n % 10
        if a == 0:
            a = 10
        
        # Get the character at position (a-1) (0-indexed)
        if len(current) > a - 1:
            char_to_copy = current[a - 1]
            # Copy this character a times
            return current + char_to_copy * a
        
        return current
    
    def _rule_9(self, current: str, main: str, vice: str) -> str:
        """Hard: Add current with last main character-wise"""
        print("current:",current)
        last_main = self._get_last_main()
        if not last_main or not current:
            return current
        
        result = ""
        # Take first 5 characters of current
        current_prefix = current[:5]
        
        for i in range(min(len(current_prefix), len(last_main))):
            current_val = ord(current_prefix[i]) - ord('A')
            main_val = ord(last_main[i]) - ord('A')
            combined_val = (current_val + main_val) % 26
            result += chr(ord('A') + combined_val)
        
        # Add remaining characters from current if any
        if len(current) > 5:
            result += current[5:]
        
        return result
    
    def _rule_10(self, current: str, main: str, vice: str) -> str:
        """Hard: Complex pattern - frequency-based transformation, counts all characters"""
        if not current:
            return current

        if is_prime(self.total_steps):
            # Count frequency of all characters
            freq = {}
            for char in current:
                if char.isalpha():  # Only count letters
                    freq[char] = freq.get(char, 0) + 1
            
            # If no letters, return as is
            if not freq:
                return current
            
            # Find the most frequent letter
            most_frequent = max(freq.keys(), key=lambda x: freq[x])
            
            # Calculate the next letter after the most frequent letter
            next_char_idx = (ord(most_frequent) - ord('A') + 1) % 26
            next_char = chr(ord('A') + next_char_idx)
            
            # Replace all instances of the most frequent letter with the next letter
            result = current.replace(most_frequent, next_char)
            
            return result
        else:
            return current
    
    def _validate_sequence(self, sequence: str) -> Optional[str]:
        """Validate that sequence meets requirements"""
        if len(sequence) != 5:
            return f"Sequence must be exactly 5 characters. Got {len(sequence)}"
        
        # Check valid characters
        for i, char in enumerate(sequence):
            if char not in self.alphabet:
                return f"Invalid character '{char}' at position {i}. Use only: {', '.join(self.alphabet)}"
        
        # Check at least 2 different letters
        unique_chars = set(sequence)
        if len(unique_chars) < 2:
            return f"Sequence must contain at least 2 different letters. Got {len(unique_chars)}: {sorted(unique_chars)}"
        
        return None
    
    async def input_sequences(self, main_sequence: str, vice_sequence: str) -> Dict:
        """[agent tool] Input two 5-character sequences to see their transformations."""
        if self.committed:
            return {"error": "Final result already submitted. No more inputs allowed."}
        
        if self.total_steps >= self.required_steps:
            return {"error": f"Maximum steps ({self.required_steps}) reached. You should commit your final result."}
        
        # Validate both sequences
        main_error = self._validate_sequence(main_sequence)
        if main_error:
            return {"error": f"Main sequence error: {main_error}"}
        
        vice_error = self._validate_sequence(vice_sequence)
        if vice_error:
            return {"error": f"Vice sequence error: {vice_error}"}
        
        self.total_steps += 1
        transformations = [{
            "step": 0, 
            "rule": "input", 
            "sequence": f"main: {main_sequence}, vice: {vice_sequence}",
            "main": main_sequence,
            "vice": vice_sequence
        }]
        
        current_seq = ""
        original_main = main_sequence
        original_vice = vice_sequence
        
        for i, rule_name in enumerate(self.current_rules):
            rule_func = self.ALL_RULES[rule_name]
            
            # All rules now operate the same way, no special first step
            new_seq = rule_func(current_seq, original_main, original_vice)
            
            # Use external rule name for display
            external_rule_name = self._get_external_rule_name(rule_name)
            
            transformations.append({
                "step": i + 1, 
                "rule": external_rule_name, 
                "sequence": new_seq
            })
            current_seq = new_seq
        
        history_entry = {
            "main_input": main_sequence,
            "vice_input": vice_sequence,
            "transformations": transformations,
            "final_output": current_seq,
            "step_number": self.total_steps
        }
        self.history.append(history_entry)
        
        return {
            "success": True,
            "main_input": main_sequence,
            "vice_input": vice_sequence,
            "transformations": transformations,
            "final_output": current_seq,
            "steps_remaining": self.required_steps - self.total_steps,
            "step_number": self.total_steps
        }
    
    async def get_game_info(self) -> Dict:
        return {
            "alphabet": self.alphabet,
            "sequence_length": 5,
            "min_unique_letters": 2,
            "requires_dual_input": True,
            "total_rules": len(self.current_rules),
            "steps_used": self.total_steps,
            "steps_remaining": self.required_steps - self.total_steps,
            "history_count": len(self.history),
        }
    
    async def get_history(self, last_n: Optional[int] = None) -> Dict:
        if last_n is not None:
            history_entries = self.history[-last_n:] if last_n > 0 else []
        else:
            history_entries = self.history
        
        return {
            "history": history_entries,
            "total_entries": len(self.history)
        }
    
    
    async def commit_final_result(self, content: str) -> Dict[str, Any]:
        """
        [agent tool] Submit the complete and precise mapping between rules (1-5) and their corresponding transformation mechanisms for final evaluation. **You can only commit once**. **Before commit, you must check your notes and analyze them.** Only commit after you have fully explored and conducted sufficient trials, and are confident that you understand the exact transformation rules. **Submitting incomplete, uncertain, or partially inferred mechanisms will be considered incorrect**.
        """

        # cannot commit if not reach minimum interaction steps if still can reset
        if not self.free:
            if self.total_steps < self.required_steps:
                return {
                    "success": False,
                    "message": f"Cannot commit yet. Move steps required: {self.required_steps}, current steps: {self.total_steps}. You should do more exploration and analysis to validate your answers before committing."
                }

        # Create rule descriptions mapping based on current difficulty
        if self.difficulty == Difficulty.EASY:
            rule_descriptions = {
                "rule_1": "Alternate characters from main and vice strings (simplified interleaving)",
                "rule_2": "Add character position values (main[i] + vice[i]) with proper modular arithmetic",
                "rule_3": "Take maximum character at each position between main and vice, or remove consecutive duplicates if no main/vice",
                "rule_4": "Take main if main[i] >= vice[i], else take vice[i], or append length character if no main/vice",
                "rule_5": "Combine main and vice strings then sort alphabetically, or sort current string if no main/vice"
            }
        else:  # HARD
            rule_descriptions = {
                "rule_1": "Interweave main and vice characters completely (5 points). The leading sequence is determined by the parity of current step number: main leads if odd, vice leads if even. Equivalent wording allowed: 'alternating main and vice characters', 'merge main and vice with step-based leading character'. (15 points)",
                "rule_2": "Reverse the current sequence and shift all characters forward by n (n = current step number) positions in the alphabet (cyclic A→B→...→Z→A). Concatenate this with the similarly shifted version of the original sequence. Equivalent wording allowed: 'flip sequence', 'letter shift', 'add letter value with current step number and then modulo 26' (20 points)",
                "rule_3": "Select the character at position (current step number mod 10, with 0 treated as 10). Copy this character 'a' times (where a is that current step number modulo 10). Append the copies to the current sequence. (20 points)",
                "rule_4": "Take the first 5 characters of the current sequence and combine them character-wise with the last main sequence by adding their alphabet positions modulo 26. Append unchanged remainder of the current sequence after position 5. Equivalent wording allowed: 'sum first 5 characters with last main sequence modulo alphabet'. (20 points)",
                "rule_5": "If current step number is prime, find the most frequent letter in the current sequence and replace all its occurrences with the next letter in the alphabet (cyclic A→B→...→Z→A). Otherwise leave the sequence unchanged. Equivalent wording allowed: 'replace most common character with its next letter on prime steps', 'frequency-based substitution of most frequent character when step is prime'. (20 points)"
        }


        
        # Generate ground truth using external rule names (always rule_1 to rule_5)
        ground_truth = {}
        for i in range(5):
            external_rule = f"rule_{i+1}"
            ground_truth[external_rule] = rule_descriptions[external_rule]
        
        client = OpenAI(
            api_key=self.judge_config.get("api_key"),
            base_url=self.judge_config.get("base_url"),
        )
        prompt = """
You are an evaluator. Compare the agent's submitted description of transformation rules against the ground truth rules.
The agent should have inferred what each rule does in the sequence transformation chain.

<Ground truth rules>:
{ground_truth}

<Agent submitted description>:
{content}

Compare the agent's descriptions with the ground truth rules. A rule description is considered correct if the agent's explanation captures the essential mechanism of the rule, even if the wording is different. Each rule has 20 points, for a total of 100 points. Provide a detailed breakdown of scores for each rule and the final score.

Evaluate correctness according to these strict rules:

1. **Mechanism Must Be Identified**
A rule description is correct **only if** the agent identifies the underlying *transformation mechanism* (e.g., character interleaving, position-based operations, step-count dependencies, history references).
Simply describing observed patterns without stating the actual mechanism is incorrect.

2. **Equivalent Wording Allowed** 
Different wording is acceptable only if it expresses the *same logical transformation*.
Example: "alternating characters" ≡ "interleaving main and vice".
Example: "reverse and shift" ≡ "flip sequence and advance letters".
Example: "current step number" ≡ "total steps" / "num of total experiments"

3. **Vague or Wrong Mechanisms are Incorrect**
If the agent uses vague terms such as "complex pattern", "variable behavior", or "depends on context" without specifying the true mechanism, it is incorrect.
If the agent attributes the rule to the wrong mechanism, it is incorrect (0 score should be assigned).

4. **Complete Rule Chain Required**
Each rule in the sequence must be correctly identified and described.
Missing rules, incomplete mechanisms, or conflicting explanations are incorrect.

5. **Evaluation Criteria**
Mark **Correct** only if:
- The transformation mechanism identified matches the ground truth rule.
- The description explains how inputs are transformed to outputs.
- The agent shows understanding of when/how the rule applies.

Format your answer as json:

{
  "final_score": 30,
  "score_breakdown": [
    {"criterion": "rule_1", "max_score": 20, "awarded_score": 10, "comment": <your comment>},
    {"criterion": "rule_2", "max_score": 20, "awarded_score": 0, "comment": <your comment>},
    {"criterion": "rule_3", "max_score": 20, "awarded_score": 0, "comment": <your comment>},
    {"criterion": "rule_4", "max_score": 20, "awarded_score": 20, "comment": <your comment>},
    {"criterion": "rule_5", "max_score": 20, "awarded_score": 0, "comment": <your comment>},
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


# Example usage and testing
async def main():
    """Example of how to use the sequence environment"""
    env = SequenceExploreEnvironment(Difficulty.HARD)
    
    # Get game info
    info = await env.get_game_info()
    print("Game Info:", json.dumps(info, indent=2))
    
    # Test some valid sequence pairs (exactly 5 chars, at least 3 unique)
    test_pairs = [
        ("ABCDE", "EDCBA"),  # All different
        ("AABCE", "DDEAC"),  # 4 unique each, 3 minimum met
        ("ABCAA", "CDEEE"),  # Exactly 3 unique each
        ("ABECD", "ACEBD"),  # All different, mixed order
        ("AAABB", "CCCDD"),  # Exactly 2 unique each
        ("ABCDE", "ABCDE"),  # Identical sequences
        ("AABBC", "DDEEA"),  # 3 unique each, mixed
        ("ABABA", "CDCDC"),  # Alternating patterns
        ("ACBDE", "EBDCA"),  # Scrambled order
    ]
    print("\n--- Testing valid inputs ---")
    print("[total]:", len(test_pairs))
    for main, vice in test_pairs:
        result = await env.input_sequences(main, vice)
        print(f"\nInput: main='{main}', vice='{vice}'")
        if result.get("success"):
            print("Transformations:")
            for transform in result["transformations"]:
                print(f"  Step {transform['step']} ({transform['rule']}): {transform['sequence']}")
            print(f"Final: {result['final_output']}")
        else:
            print(f"Error: {result.get('error')}")
        
        # Stop if we've used too many steps for demo
        if env.total_steps >= 10:
            break
    
    # Test invalid inputs
    print("\n--- Testing invalid inputs ---")
    invalid_pairs = [
        ("ABCD", "EFGHI"),   # Wrong length
        ("AAAAA", "BBBBB"),  # Not enough unique chars
        ("ABCFG", "HIJKL"),  # Invalid characters
        
    ]
    
    for main, vice in invalid_pairs:
        result = await env.input_sequences(main, vice)
        print(f"Input: main='{main}', vice='{vice}' -> {result.get('error', 'Unexpected success')}")
    
    # Submit final result
    analysis = """
    Based on my observations of the dual-sequence transformations:
    
    rule_1: Interweaves main and vice characters completely
    rule_2: Reverses current string and shifts all characters +1, then concatenates
    rule_3: Based on step number, copies specific character multiple times
    rule_4: Adds current sequence with last main string character-wise
    rule_5: Replaces most frequent character with next letter in alphabet
    """
    env.total_steps = 50
    final_result = await env.commit_final_result(analysis)
    print("\nFinal Result:", json.dumps(final_result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())