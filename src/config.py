import yaml
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path

# Helper function to recursively convert dict to nested dataclasses
def _from_dict(data_class, data):
    if not isinstance(data, dict):
        return data
    field_types = {f.name: f.type for f in data_class.__dataclass_fields__.values()}
    return data_class(**{
        key: (_from_dict(field_types[key], value) if hasattr(field_types.get(key), '__dataclass_fields__') else value)
        for key, value in data.items()
        if key in field_types
    })

# Define Dataclasses matching the YAML structure

@dataclass
class SLMSpecialTokensConfig:
    assistant_thought_begin: str = "<assistant_thought>"
    assistant_thought_end: str = "</assistant_thought>"

@dataclass
class SLMConfig:
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    context_length: int = 131072
    generation_max_length: int = 512
    quantization: str = "4-bit"
    special_tokens: SLMSpecialTokensConfig = field(default_factory=SLMSpecialTokensConfig)

@dataclass
class PromptSeedsConfig:
    name: str = "wikitext"
    subset: str = "wikitext-2-raw-v1"
    split: str = "train"
    min_prompt_words: int = 20
    cache_path: str = "data/prompts.json"
    sentences_cache_path: str = "data/prompts_sentences.jsonl"

@dataclass
class TopicDriftEmbeddingsConfig:
    model: str = "text-embedding-3-large"
    cache_path: str = "assets/embeds.npy"
    embed_tasks_path: str = "data/embed_tasks.jsonl"

@dataclass
class QAEvalConfig:
    squad: str = "squad"
    boolq: str = "boolq"

@dataclass
class DatasetsConfig:
    prompt_seeds: PromptSeedsConfig = field(default_factory=PromptSeedsConfig)
    topic_drift_embeddings: TopicDriftEmbeddingsConfig = field(default_factory=TopicDriftEmbeddingsConfig)
    qa_eval: QAEvalConfig = field(default_factory=QAEvalConfig)

@dataclass
class FeatureToggleConfig:
    enabled: bool = True

@dataclass
class ZEntropyConfig(FeatureToggleConfig):
    window_size: int = 16

@dataclass
class ContextVecConfig(FeatureToggleConfig):
    hidden_size: int = 3072

@dataclass
class FeaturesConfig:
    entropy: FeatureToggleConfig = field(default_factory=FeatureToggleConfig)
    logit_gap: FeatureToggleConfig = field(default_factory=FeatureToggleConfig)
    z_entropy: ZEntropyConfig = field(default_factory=ZEntropyConfig)
    topic_drift: FeatureToggleConfig = field(default_factory=FeatureToggleConfig)
    pos_in_prompt: FeatureToggleConfig = field(default_factory=FeatureToggleConfig)
    in_thought_flag: FeatureToggleConfig = field(default_factory=FeatureToggleConfig)
    context_vec: ContextVecConfig = field(default_factory=ContextVecConfig)

@dataclass
class ActionConfig:
    id: int

@dataclass
class BranchActionConfig(ActionConfig):
    k: int = 3

@dataclass
class TempBumpActionConfig(ActionConfig):
    factor: float = 1.3

@dataclass
class MacroActionsConfig:
    argmax: ActionConfig = field(default_factory=lambda: ActionConfig(id=0))
    branch: BranchActionConfig = field(default_factory=lambda: BranchActionConfig(id=1, k=3))
    resample: ActionConfig = field(default_factory=lambda: ActionConfig(id=2))
    think_begin: ActionConfig = field(default_factory=lambda: ActionConfig(id=3))
    think_end: ActionConfig = field(default_factory=lambda: ActionConfig(id=4))
    temperature_bump: TempBumpActionConfig = field(default_factory=lambda: TempBumpActionConfig(id=5, factor=1.3))

@dataclass
class MCTSConfig:
    simulations_per_step: int = 8
    ucb_c: float = 1.5
    rollout_max_length: int = 32
    simulations_epoch_10_plus: int = 16

@dataclass
class SelfPlayRewardConfig:
    coherence_weight: float = 1.0
    length_penalty_weight: float = 0.0

@dataclass
class SelfPlayConfig:
    initial_prompt_max_length: int = 64
    episode_max_tokens: int = 256
    reward: SelfPlayRewardConfig = field(default_factory=SelfPlayRewardConfig)

@dataclass
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 1.0e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    entropy_bonus_weight: float = 0.01
    grad_clip_norm: float = 1.0
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingLR"
    epochs: int = 20
    save_every_seconds: int = 7200

@dataclass
class LongRunConfig:
    episodes: int = 5000
    simulations: int = 8

@dataclass
class KenLMConfig:
    model_path: str = "assets/wiki_5gram.arpa"
    auto_download: bool = True
    placeholder_score: float = 0.0

@dataclass
class OpenAIConfig:
    # API key is expected via environment variable OPENAI_API_KEY
    pass


@dataclass
class AppConfig:
    slm: SLMConfig = field(default_factory=SLMConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    macro_actions: MacroActionsConfig = field(default_factory=MacroActionsConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    long_run: LongRunConfig = field(default_factory=LongRunConfig)
    kenlm: KenLMConfig = field(default_factory=KenLMConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    random_seed: int = 42
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, config_path: Path = Path("configs/default.yaml")) -> "AppConfig":
        try:
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            return _from_dict(cls, yaml_data)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default AppConfig values.")
            return cls() # Return a default instance
        except Exception as e:
            print(f"Warning: Error loading config from {config_path}: {e}. Using default AppConfig values.")
            return cls() # Return a default instance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Global config instance, loaded once
try:
    cfg = AppConfig.from_yaml()
except FileNotFoundError:
    print("Config file not found. Using default values.")
    cfg = AppConfig()
except Exception as e:
    print(f"Error loading config: {e}. Using default values.")
    cfg = AppConfig()

if __name__ == "__main__":
    # Example of how to access config values
    print(f"SLM Model Name: {cfg.slm.model_name}")
    print(f"MCTS Simulations: {cfg.mcts.simulations_per_step}")
    print(f"Log Level: {cfg.log_level}")
    # print(cfg.to_dict()) 