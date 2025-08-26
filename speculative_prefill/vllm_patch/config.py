import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import yaml


@dataclass
class SpecConfig:
    keep_strategy: Optional[str]
    keep_kwargs: Optional[Dict[str, Any]] = None
    look_ahead_cnt: int = 8
    pool_kernel_size: Optional[int] = None
    ignore_eos: bool = False # for benchmarking only ideally

    @classmethod
    def from_path(cls, config_path: Optional[str] = None):
        if config_path is None:
            # Return default config when no path provided
            return cls(keep_strategy="percentage", keep_kwargs={"percentage": 0.5})

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Get the fields of the dataclass
        field_names = {f.name for f in fields(cls)}
        
        # Check for unused fields in the YAML
        unused_fields = set(data.keys()) - field_names
        if unused_fields:
            raise ValueError(f"Unused fields in YAML: {unused_fields}.")
        
        # Filter out keys that are not fields of the dataclass
        used_data = {k: v for k, v in data.items() if k in field_names}
        
        # Return an instance of the dataclass populated with the filtered data
        return cls(**used_data)
    
    def __post_init__(self):
        assert self.keep_strategy in ["percentage"]

        if self.keep_strategy is None:
            self.keep_strategy = "percentage"
            self.keep_kwargs["percentage"] = 0.5

        if self.keep_kwargs is None:
            self.keep_kwargs = {}


_SPEC_CONFIG: Optional[SpecConfig] = None

def init_spec_config():
    global _SPEC_CONFIG
    if _SPEC_CONFIG is None:
        _SPEC_CONFIG = SpecConfig.from_path(
            os.environ.get("SPEC_CONFIG_PATH")
        )
        print("\033[92m{}\033[00m".format(
            f"Using spec config:\n{json.dumps(asdict(_SPEC_CONFIG), indent=4)}"))

def get_spec_config() -> SpecConfig:
    global _SPEC_CONFIG
    assert _SPEC_CONFIG is not None
    return _SPEC_CONFIG