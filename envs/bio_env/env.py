import random
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import itertools
import asyncio
from enum import Enum
from typing import Dict, List, Tuple, Any, Callable
import json
import openai
from openai import OpenAI
from envs.common import Difficulty

ENV_PROMPT='''
=== ALIEN ORGANISM GENETICS LABORATORY ===

TASK OVERVIEW:
You are a researcher in a laboratory studying a newly discovered alien organism species. Each organism has multiple traits controlled by hidden genetic factors. Your goal is to discover the rules that govern inheritance and expression of these traits.

BASIC RULES:
- You can perform controlled crosses between organisms to produce offspring.
- Each cross consumes 1 experimental step. There is a required total steps of {required_steps} to complete the game.
- Each experiment yields a number of offspring with observable traits (phenotypes).
- Some crosses may produce non-viable offspring (lethal combinations).
- You can query organisms in the lab to examine their traits, descriptions, and lineage.
- There is a maximum number of experiments you can perform.
- You have a minimum number of steps required before submitting your final conclusions.

EXPLORATION GOALS:
- Determine how traits are inherited across generations.
- Identify patterns of dominance, interaction, or dosage effects in traits.
- Detect lethal combinations that prevent offspring viability.
- Develop a predictive model for offspring traits from parental organisms.
- Record observations, generate hypotheses, and refine understanding through repeated experiments.

AVAILABLE ACTIONS (agent tools):
1. `conduct_cross(parent1_id, parent2_id, num_offspring)`:
   - Cross two organisms to produce offspring.
   - Returns offspring phenotypes, viability rates, and lethality information.
2. `query_organisms(start_id, end_id, include_genotype=False)`:
   - Examine organisms in a given ID range.
   - Returns phenotypes, generation, parents, description, and optionally genotypes.
3. `get_lab_status()`:
   - Provides laboratory resource usage, experiment progress, and remaining experiments.
4. `commit_final_result(content)`:
   - Submit final conclusions describing inferred inheritance rules.
   - Can only submit once after the minimum steps are met.

IMPORTANT NOTES:
- Hidden mechanisms may include complex inheritance patterns, dominance, dosage effects, interactions, and lethal combinations.
- Effects are consistent but not revealed upfront; repeated experiments and careful observation are required.
- Non-viable offspring are part of the system and provide critical information.
- Organisms may interact in non-obvious ways; consider lineage and trait combinations.
- Record your observations carefully. Your final submission should explain the mechanisms you inferred and provide predictions for untested crosses.


COMMIT GUIDELINES:
Please produce a **formal experimental report**. The report must strictly follow the structure below and fully expand each section:

1. **Genetic Basis**
- Describe in detail the chromosomal characteristics of the organism (e.g., number of chromosome sets).
- Explain the mechanisms of gamete production: how gametes are formed, whether special segregation rules exist.

2. **Inheritance Rules for Each Trait**
- For each trait, clearly list the exact number of alleles and describe the effect of each allele.
- Explain the dominance relationships (dominant, recessive, dosage effects, etc.).
- State any special mechanisms (e.g., position effects, dosage sensitivity).

3. **Lethal Combinations and Special Cases**
- Enumerate all discovered lethal genotypes or non-viable combinations, and explain why they are lethal.
- Highlight any exceptions, unusual cases, or conditional outcomes.

**Writing Requirements:**

- Use clear, hierarchical section headings.
- The report must be formal and logically consistent.
- The explanation must be fully self-contained, such that the report alone can explain all experimental results.
'''

class GeneticsLabEnvironment:
    """
    Triploid Alien Organism Genetics Laboratory Environment
    
    Core Features:
    1. Triploid organisms: each gene locus has 3 alleles
    2. Special meiosis mechanism: 3 chromosomes undergo unequal segregation
    3. Complex gene expression patterns: dosage effects, dominance hierarchy, cyclic interactions
    4. Limited experimental resources: simulates real research constraints
    5. Agent evaluation: tests exploration, reasoning, and planning abilities
    """
    
    def __init__(self, seed: int = 42, required_steps: int = 50, 
                 difficulty: Difficulty = Difficulty.HARD, free=False):
        random.seed(seed)
        self.seed = seed
        self.difficulty = difficulty
        self.env_prompt = ENV_PROMPT
        
        # Gene and allele definitions - only 3 traits
        self.genes = ['body_size', 'color', 'shell_shape']
        self.alleles = {
            'body_size': ['S1', 'S2', 'S3'],      # Size variants (dosage effect)
            'color': ['C1', 'C2', 'C3'],          # Color variants (dominance hierarchy)
            'shell_shape': ['H1', 'H2', 'H3']     # Shell shape variants (cyclic interaction + lethal)
        }
        
        # Experimental resource constraints
        self.required_steps = required_steps
        self.free = free
        self.current_experiments = 0
        self.max_organisms = 200
        
        # Organism storage
        self.organisms = {}
        self.next_id = 1
        
        # Experiment tracking
        self.experiment_log = []
        self.discovered_rules = {}
        self.committed = False
        self.final_result = {}
        
        # Hidden genetic rules (for agent to discover)
        self._setup_hidden_rules()
        
        # Initialize laboratory stock
        self._initialize_lab_stock()

        self.judge_config = self.load_judge_config()

        
        if free:
            self.env_prompt = ENV_PROMPT.replace(" There is a required total steps of {required_steps} to complete the game.","")
        else:
            self.env_prompt = ENV_PROMPT.format(required_steps=self.required_steps)

        print("ENV PROMPT:\n", self.env_prompt)
    
    def noisy(self, value: float, noise_level: float = 0.05) -> float:
        """Add noise to a value for realism"""
        noise = value * noise_level * (random.random() - 0.5) * 2
        return round(max(0, value + noise),2)

    def load_judge_config(self):
        """Load judge model configuration from a YAML file"""
        import yaml
        config_path = 'judge_config.yaml' #os.path.join(os.path.dirname(__file__), 'judge_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Judge config loaded:", config)
        return config
        
    def _setup_hidden_rules(self):
        """Setup hidden genetic inheritance rules"""
        
        # 1. Meiosis rule: 3 chromosomes -> unequal segregation (1+2)
        self.meiosis_rule = "unequal_segregation"
        
        # 2. Gene expression patterns - only 3 traits
        self.expression_rules = {
            'body_size': {
                'type': 'dosage_effect',  # Dosage effect - additive
                'rule': {
                    'S1': {'size_value': 200, 'description': 'large'},
                    'S2': {'size_value': 50, 'description': 'medium'}, 
                    'S3': {'size_value': 10, 'description': 'small'}
                }
            },
            'color': {
                'type': 'dominance_hierarchy',  # Dominance hierarchy
                'rule': ['C1', 'C2', 'C3'],  # C1 > C2 > C3
                'phenotypes': {
                    'C1': {'color': 'red', 'intensity': 95},
                    'C2': {'color': 'blue', 'intensity': 60}, 
                    'C3': {'color': 'white', 'intensity': 20}
                }
            },
            'shell_shape': {
                'type': 'cyclic_interaction_lethal',  # Cyclic interaction with lethality
                'rule': {
                    'H1': {'beats': 'H2', 'loses_to': 'H3'},
                    'H2': {'beats': 'H3', 'loses_to': 'H1'},
                    'H3': {'beats': 'H1', 'loses_to': 'H2'}
                },
                'lethal_combination': True,  # All three present = lethal
                'shell_shapes': {
                    'H1': {'shape': 'spiky', 'hardness': 100},
                    'H2': {'shape': 'smooth', 'hardness': 100},
                    'H3': {'shape': 'ridged', 'hardness': 100}
                }
            }
        }
    
    def _initialize_lab_stock(self):
        """Initialize laboratory with pure breeding lines"""
        
        # Three homozygous lines as starting material
        initial_organisms = [
            {
                'genotype': {
                    'body_size': ('S1', 'S1', 'S2'), 
                    'color': ('C1', 'C1', 'C1'), 
                    'shell_shape': ('H1', 'H1', 'H1')
                },
                'description': 'Initial Organism (Line A)'
            },
            {
                'genotype': {
                    'body_size': ('S1', 'S2', 'S2'), 
                    'color': ('C2', 'C2', 'C2'), 
                    'shell_shape': ('H2', 'H2', 'H2')
                }, 
                'description': 'Initial Organism (Line B)'
            },
            {
                'genotype': {
                    'body_size': ('S3', 'S3', 'S3'), 
                    'color': ('C3', 'C3', 'C3'), 
                    'shell_shape': ('H3', 'H3', 'H3')
                },
                'description': 'Initial Organism (Line C)'
            }
        ]
        
        for org_data in initial_organisms:
            self._create_organism(org_data['genotype'], org_data['description'])
    
    def _create_organism(self, genotype: Dict[str, Tuple[str, str, str]], description: str = "") -> Optional[int]:
        """Create new organism instance - now only checks lab capacity"""
        if len(self.organisms) >= self.max_organisms:
            raise Exception("Laboratory organism capacity reached!")
            
        # Removed lethality check here, all lethality checks are performed during crossbreeding
        organism_id = self.next_id
        phenotype = self._calculate_phenotype(genotype)
        
        self.organisms[organism_id] = {
            'id': organism_id,
            'genotype': genotype,
            'phenotype': phenotype,
            'generation': getattr(self, '_current_generation', 0),
            'parents': getattr(self, '_current_parents', None),
            'description': description,
            'viable': True
        }
        
        self.next_id += 1
        return organism_id
    
    def _is_viable_genotype(self, genotype: Dict[str, Tuple[str, str, str]]) -> bool:
        """Unified genotype viability check: check shell gene lethal combinations"""
        shell_alleles = set(genotype['shell_shape'])
        # If shell gene contains all three alleles H1, H2, H3 simultaneously, it is lethal
        return len(shell_alleles) < 3
    
    def _calculate_phenotype(self, genotype: Dict[str, Tuple[str, str, str]], ) -> Dict[str, Any]:
        """Calculate phenotype from genotype"""
        phenotype = {}
        
        for gene, alleles in genotype.items():
            if gene == 'body_size':
                phenotype['body_size'], phenotype['size_score'] = self._calculate_size_phenotype(alleles)
            elif gene == 'color':
                phenotype['body_color'], phenotype['color_intensity'] = self._calculate_color_phenotype(alleles)
            elif gene == 'shell_shape':
                phenotype['shell_shape'], phenotype['shell_hardness'] = self._calculate_shell_phenotype(alleles)
        
        return phenotype
    
    def _calculate_size_phenotype(self, alleles: Tuple[str, str, str]) -> Tuple[str, int]:
        """Calculate size phenotype (dosage effect)"""
        #  [30, 70, 110, 120, 150, 160, 200, 210, 250, 300] 
        rule = self.expression_rules['body_size']['rule']
        
        # Sum up all allele contributions (additive effect)
        total_size = sum(rule[allele]['size_value'] for allele in alleles)
        total_size = self.noisy(total_size, noise_level=0.01)
        
        # Determine size category
        if total_size >= 250:
            return "extra_large", total_size
        elif total_size >= 180:
            return "large", total_size
        elif total_size >= 100:
            return "medium", total_size
        elif total_size >= 50:
            return "small", total_size
        else:
            return "tiny", total_size
    
    def _calculate_color_phenotype(self, alleles: Tuple[str, str, str]) -> Tuple[str, int]:
        """Calculate color phenotype (dominance hierarchy)"""
        rule = self.expression_rules['color']
        hierarchy = rule['rule']
        phenotypes = rule['phenotypes']
        
        # Find most dominant allele present
        for dominant_allele in hierarchy:
            if dominant_allele in alleles:
                pheno = phenotypes[dominant_allele]
                return pheno['color'], pheno['intensity']
        
        return "colorless", 0
    
    def _calculate_shell_phenotype(self, alleles: Tuple[str, str, str]) -> Tuple[str, int]:
        """Calculate shell phenotype (cyclic interaction)"""
        rule = self.expression_rules['shell_shape']['rule']
        shell_shapes = self.expression_rules['shell_shape']['shell_shapes']
        
        allele_set = set(alleles)
        allele_counts = Counter(alleles)
        
        # If all three types present, it's lethal (shouldn't reach here due to early filtering)
        if len(allele_set) == 3:
            return "deformed", 0
        
        # Single allele type
        if len(allele_set) == 1:
            allele = list(allele_set)[0]
            shape_info = shell_shapes[allele]
            return shape_info['shape'], self.noisy(shape_info['hardness'], noise_level=0.05)
        
        # Two allele types - determine winner
        allele_list = list(allele_set)
        winner = None
        
        for allele in allele_list:
            other_allele = [a for a in allele_list if a != allele][0]
            if rule[allele]['beats'] == other_allele:
                winner = allele
                break
        
        if winner:
            # Winner dominates, but strength depends on count
            count_ratio = allele_counts[winner] / 3.0
            base_hardness = shell_shapes[winner]['hardness']
            actual_hardness = int(base_hardness * count_ratio)
            return shell_shapes[winner]['shape'], self.noisy(actual_hardness, noise_level=0.05)
        
        return "mixed", self.noisy(actual_hardness, noise_level=0.05)  # Tie situation
    
    def _generate_gametes(self, genotype: Dict[str, Tuple[str, str, str]]) -> List[Dict[str, Tuple[str, ...]]]:
        """Generate gametes through triploid meiosis (unequal segregation)"""
        # print("Generating gametes for genotype:", genotype)
        gametes = []
        
        # For each gene, perform meiotic division
        for gene in self.genes:
            alleles = list(genotype[gene])
            
            # Triploid meiosis: 3 chromosomes segregate as 1+2
            # Randomly assign each chromosome to gamete 1 or 2
            assignments = [0, 1, 1]  # 0 = gamete1, 1 = gamete2
            random.shuffle(assignments)
            
            gamete1_alleles = []
            gamete2_alleles = []
            
            for i, assignment in enumerate(assignments):
                if assignment == 0:
                    gamete1_alleles.append(alleles[i])
                else:
                    gamete2_alleles.append(alleles[i])
            
            # Store gamete compositions
            if not gametes:
                gametes = [
                    {gene: tuple(gamete1_alleles)},
                    {gene: tuple(gamete2_alleles)}
                ]
            else:
                gametes[0][gene] = tuple(gamete1_alleles)
                gametes[1][gene] = tuple(gamete2_alleles)
            
            # print("gametes:", gametes)
        
        return gametes
    
    def _remove_organisms(self, organism_ids: List[int]):
        """Remove organisms from the lab due to lab volume constraints"""
        for org_id in organism_ids:
            if org_id in self.organisms:
                del self.organisms[org_id]
    
    # ==== Agent Interface Methods ====
    
    async def conduct_cross(self, parent1_id: int, parent2_id: int, num_offspring: int = 10) -> Dict[str, Any]:
        """[agent tool] Conduct breeding experiment between two organisms to produce offspring"""
        
        if self.committed:
            return {"success": False, "message": "Experiment concluded - no further experiments allowed"}
            
        if self.current_experiments >= self.required_steps:
            return {"success": False, "message": f"Maximum experiment limit reached ({self.required_steps}). You should commit your answer"}
        
        if parent1_id not in self.organisms or parent2_id not in self.organisms:
            return {"success": False, "message": "One or both parent organisms not found"}
        
        if num_offspring <= 0 or num_offspring > 100:
            return {"success": False, "message": "Number of offspring must be between 1 and 100"}
        
        if len(self.organisms) + num_offspring > self.max_organisms:
            return {"success": False, "message": "Laboratory organism capacity will be exceeded. Remove some organisms first."}
        
        parent1 = self.organisms[parent1_id]
        parent2 = self.organisms[parent2_id]
        
        # Generate gametes from both parents
        
        
        offspring_data = []  # List of {id, phenotype} pairs
        lethal_count = 0
        fertilization_attempts = 0
        
        self._current_generation = max(parent1.get('generation', 0), parent2.get('generation', 0)) + 1
        self._current_parents = (parent1_id, parent2_id)
        
        # Continue until we get the required number of viable offspring
        while len(offspring_data) < num_offspring:
            fertilization_attempts += 1

            gametes1 = self._generate_gametes(parent1['genotype'])
            gametes2 = self._generate_gametes(parent2['genotype'])
            
            # Random fertilization
            gamete1 = random.choice(gametes1)
            gamete2 = random.choice(gametes2)
            
            # Form zygote genotype
            offspring_genotype = {}
            is_viable = True
            
            for gene in self.genes:
                # Combine alleles from both gametes
                combined_alleles = list(gamete1[gene]) + list(gamete2[gene])
                
                # Lethality check 1: check if triploid
                if len(combined_alleles) != 3:
                    is_viable = False
                    break
                
                offspring_genotype[gene] = tuple(sorted(combined_alleles))
            
            # Lethality check 2: check if shell gene contains all three allele types
            if is_viable and not self._is_viable_genotype(offspring_genotype):
                is_viable = False
            
            if not is_viable:
                lethal_count += 1
                continue
            # print("fertilization_attempts:", fertilization_attempts)
            
            # Create viable organism
            offspring_id = self._create_organism(
                offspring_genotype, 
                f"Cross {parent1_id} × {parent2_id} offspring #{len(offspring_data)+1}"
            )
            
            # Since we've checked all lethal conditions during crossbreeding, this should not return None
            if offspring_id is not None:
                offspring_data.append({
                    "id": offspring_id,
                    "phenotype": self.organisms[offspring_id]['phenotype']
                })
            
            # Safety check to prevent infinite loops
            if fertilization_attempts > num_offspring * 100:  # Arbitrary large multiplier
                return {
                    "success": False, 
                    "message": f"Unable to produce {num_offspring} viable offspring after {fertilization_attempts} attempts. This cross may have very low viability."
                }
        
        # Record experiment
        self.current_experiments += 1
        experiment_record = {
            'experiment_id': self.current_experiments,
            'parent1_id': parent1_id,
            'parent2_id': parent2_id,
            'parent1_phenotype': parent1['phenotype'],
            'parent2_phenotype': parent2['phenotype'],
            'requested_offspring': num_offspring,
            'viable_offspring_data': offspring_data,
            'lethal_count': lethal_count,
            'total_fertilization_attempts': fertilization_attempts,
            'viability_rate': len(offspring_data) / fertilization_attempts if fertilization_attempts > 0 else 0
        }
        self.experiment_log.append(experiment_record)
        
        return {
            "success": True,
            "experiment_id": self.current_experiments,
            "parent1_id": parent1_id,
            "parent2_id": parent2_id,
            "parent1_phenotype": parent1['phenotype'],
            "parent2_phenotype": parent2['phenotype'],
            "requested_offspring": num_offspring,
            "viable_offspring_count": len(offspring_data),
            "lethal_offspring_count": lethal_count,
            "total_fertilization_attempts": fertilization_attempts,
            "viability_rate": round(len(offspring_data) / fertilization_attempts if fertilization_attempts > 0 else 0, 3),
            "offspring": offspring_data,
            "remaining_experiments": self.required_steps - self.current_experiments,
            "total_organisms": len(self.organisms)
        }
    
    async def query_organisms(self, start_id: int = 1, end_id: Optional[int] = None) -> Dict[str, Any]:
        """[agent tool] Query organisms in the laboratory by ID range to examine their traits"""
        
        if end_id is None:
            end_id = max(self.organisms.keys()) if self.organisms else start_id
            
        if start_id < 1 or (end_id is not None and end_id < start_id):
            return {"success": False, "message": "Invalid ID range specified"}
        
        organisms_found = []
        for org_id in range(start_id, end_id + 1):
            if org_id in self.organisms:
                org_info = {
                    'id': org_id,
                    'phenotype': self.organisms[org_id]['phenotype'],
                    'generation': self.organisms[org_id]['generation'],
                    'parents': self.organisms[org_id]['parents'],
                    'description': self.organisms[org_id]['description'],
                    'viable': self.organisms[org_id]['viable']
                }
                
                    
                organisms_found.append(org_info)
        
        return {
            "success": True,
            "message": f"Found {len(organisms_found)} organisms in range {start_id}-{end_id}",
            "query_range": f"{start_id}-{end_id}",
            "organisms_found": len(organisms_found),
            "organisms": organisms_found,
            "total_lab_organisms": len(self.organisms)
        }
       
    async def remove_organisms(self, organism_ids: List[int]) -> Dict[str, Any]:
        """[agent tool] Remove specified organisms from the laboratory to manage capacity"""
        
        if self.committed:
            return {"success": False, "message": "Experiment concluded - no further modifications allowed"}
        
        removed_count = 0
        for org_id in organism_ids:
            if org_id in self.organisms:
                del self.organisms[org_id]
                removed_count += 1
        
        return {
            "success": True,
            "message": f"Removed {removed_count} organisms from the laboratory",
            "requested_removals": len(organism_ids),
            "actual_removals": removed_count,
            "total_lab_organisms": len(self.organisms)
        }

    async def get_lab_status(self) -> Dict[str, Any]:
        """[agent tool] Get current laboratory status including resource usage and experiment progress"""
        
        return {
            "success": True,
            "message": "Laboratory status retrieved successfully",
            "current_experiments": self.current_experiments,
            "remaining_experiments": self.required_steps - self.current_experiments,
            "can_commit": self.current_experiments >= self.required_steps,
            "total_organisms": len(self.organisms),
            "max_organisms": self.max_organisms,
            "experiment_log_entries": len(self.experiment_log),
            "committed": self.committed,
        }
    
    async def commit_final_result(self, content: str) -> Dict[str, Any]:
        """
        [agent tool] Submit the final inferred genetic inheritance rules for evaluation.
        You can only submit once. Submit only after sufficient experimentation and analysis.
        Submissions must include all five inference targets:
        1. Ploidy level
        2. Meiotic segregation mechanism
        3. Trait A (body size) inheritance rule
        4. Trait B (color) inheritance rule
        5. Trait C (shell shape) inheritance rule
        """

        if not self.free:
            if self.current_experiments < self.required_steps:
                return {
                    "success": False,
                    "message": f"Cannot commit yet. Experiments required: {self.required_steps}, current: {self.current_experiments}. Conduct more experiments before committing."
                }

    
        # Build evaluation prompt
        client = OpenAI(
            api_key=self.judge_config.get("api_key"),
            base_url=self.judge_config.get("base_url"),
        )
        prompt = """
You are an evaluator. Compare the agent's submitted description of inheritance rules against the scoring rules.

<Scoring rules>:
## A. Fundamental Understanding (25 points)
1. Recognition of triploidy - Identifies that the organism is triploid (three alleles per locus). *(Medium, 15 points)*
2. Meiosis process - Correctly describes the special meiosis mechanism (gametes are 1n and 2n due to unequal segregation). *(Easy, 5 points)*
3. Viability constraint - Explains that only triploid zygotes are viable; other ploidy combinations are lethal. *(Easy, 5 points)*

## B. Body Size Trait (Dosage Effect) (35 points)
4. Inheritance rule - Discovers that body size follows an additive dosage effect. *(Easy, 5 points)*
5. Allele identification - Correctly identifies the existence of three distinct alleles for size. *(Medium, 10 points)*
6. Quantitative description - Provides accurate quantitative values for the three alleles (approx. 200 ± 20; 50 ± 10; 10 ± 5). *(Hard, 20 points)*


## C. Color Trait (Dominance Hierarchy) (10 points)
7. Dominance hierarchy - Identifies that color follows a strict dominance order: Red (C1) > Blue (C2) > White (C3). *(Easy, 5 points)*
8. Complete dominance - States that the dominance is complete: phenotype intensity is determined solely by the dominant allele. *(Easy, 5 points)*


## D. Shell Trait (Cyclic Interaction & Lethality) (30 points)
9. Cyclic dominance - Identifies that shell shape alleles follow a cyclic dominance relationship: Spiky (H1) > Smooth (H2) > Ridged (H3) > Spiky. *(Medium, 10 points)*
10. Lethal combination - Explains that the presence of all three shell alleles (H1 + H2 + H3) is lethal. *(Hard, 20 points)*


<Agent submitted report>:
{content}

Evaluation guidelines:

1. **Mechanism Identification Required**
For each item, award the full score if the point is clearly and completely correctly addressed; partial credit **cannot** be given if the description is incomplete, vague, or not correct.

2. **Equivalent Wording Allowed**
Different wording is correct if logically equivalent.
- "3 gene copies" ≡ "triploid"
- "1+2 split" ≡ "unequal segregation"
- "dosage/additive effect" ≡ "sum of allele contributions"
- "dominance ladder" ≡ "dominance hierarchy"
- "rock-paper-scissors interaction" ≡ "cyclic dominance"

3. **Incomplete or Vague = Incorrect**
Descriptions like "complex" or "varies" without identifying the true mechanism are incorrect.

Format your answer as json:

{
  "final_score": 87,
  "score_breakdown": [
    {"criterion": "Triploidy recognition", "max_score": 15, "awarded_score": 0, "comment": <your comment>},
    {"criterion": "Meiosis process (1n/2n gametes)", "max_score": 5, "awarded_score": 5, "comment": <your comment>},
    {"criterion": "Viability constraint (only triploid survives)", "max_score": 5, "awarded_score": 5, "comment": <your comment>},
    {"criterion": "Body size: dosage effect", "max_score": 5, "awarded_score": 5, "comment": <your comment>},
    {"criterion": "Body size: allele identification", "max_score": 10, "awarded_score": 10, "comment": <your comment>},
    {"criterion": "Body size: quantitative values", "max_score": 20, "awarded_score": 15, "comment": <your comment>},
    {"criterion": "Color: dominance hierarchy", "max_score": 5, "awarded_score": 5, "comment": <your comment>},
    {"criterion": "Color: complete dominance", "max_score": 5, "awarded_score": 5, "comment": <your comment>},
    {"criterion": "Shell: cyclic dominance", "max_score": 10, "awarded_score": 0, "comment": <your comment>},
    {"criterion": "Shell: lethal combination", "max_score": 20, "awarded_score": 0, "comment": <your comment>}
  ]
}
"""
        prompt = prompt.replace("{content}", content)
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
    
    
    def get_ground_truth(self) -> Dict[str, Any]:
        """Reveal the true genetic rules for evaluation purposes"""
        return {
            'meiosis_mechanism': 'unequal_segregation_1_plus_2',
            'body_size_rule': 'dosage_effect_additive',
            'color_rule': 'dominance_hierarchy_C1_C2_C3',
            'shell_shape_rule': 'cyclic_interaction_with_lethality',
            'lethal_combination': 'all_three_shell_alleles_present',
            'detailed_rules': self.expression_rules
        }


# Demo function for testing
async def demo_triploid_lab():
    """Demonstrate the Triploid Genetics Lab environment"""
    print("=== Triploid Genetics Laboratory Demo ===\n")
    
    # Initialize lab
    lab = GeneticsLabEnvironment()
    
    print("1. Initial laboratory organisms:")
    # initial_query = await lab.query_organisms(1, 3)
    # for org in initial_query['organisms']:
    #     print(f"   Organism {org['id']}: {org['phenotype']}")
    #     print(f"      Description: {org['description']}")
    
    lab.current_experiments = 5
    output = await lab.commit_final_result("No pattern found")
    # print(output)
    # input("?    ")
    lab.max_organisms = 10000000
    
    # First cross: Line A × Line B
    result1 = await lab.conduct_cross(1, 2, num_offspring=100)
    print(result1)
    result2 = await lab.conduct_cross(1, 1, num_offspring=100)
    print(result2)
    input("-"*20)
    input("Press Enter to continue...")
    print(f"   Cross 1 (A × B): {result1['viable_offspring_count']} viable offspring")
    print(f"   Lethality rate: {result1['lethal_offspring_count']}/15 = {(result1['lethal_offspring_count']/15)*100:.1f}%")
    
    # Second cross: Line B × Line C  
    result2 = await lab.conduct_cross(2, 3, num_offspring=200)
    print(result2)
    print(f"   Cross 2 (B × C): {result2['viable_offspring_count']} viable offspring")
    print(f"   Lethality rate: {result2['lethal_offspring_count']}/15 = {(result2['lethal_offspring_count']/15)*100:.1f}%")
    
    # Third cross: Line A × Line C
    result3 = await lab.conduct_cross(1, 3, num_offspring=200)
    print(f"   Cross 3 (A × C): {result3['viable_offspring_count']} viable offspring")
    print(f"   Lethality rate: {result3['lethal_offspring_count']}/15 = {(result3['lethal_offspring_count']/15)*100:.1f}%")
    





if __name__ == "__main__":
    asyncio.run(demo_triploid_lab())
    # test_deadlock_scenarios()