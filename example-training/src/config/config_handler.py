import yaml
from typing import Dict, Any, List


class ConfigHandler:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration"""
        return self.config
    
    def to_markdown_tables(self, exclude_sections: List[str] = None) -> str:
        """Convert configuration to markdown tables"""
        if exclude_sections is None:
            exclude_sections = ["output", "conversion", "huggingface", "tracking"]
        else:
            exclude_sections = exclude_sections + ["output", "conversion", "huggingface", "tracking"]
        
        markdown = ""
        for section, section_data in self.config.items():
            if section in exclude_sections:
                continue
            markdown += f"## {section.title()} Configuration\n"
            markdown += "| Parameter | Value |\n"
            markdown += "|-----------|-------|\n"
            
            for param, value in section_data.items():
                if isinstance(value, (list, dict)):
                    formatted_value = str(value)
                else:
                    formatted_value = str(value)     
                markdown += f"| {param} | {formatted_value} |\n"
            markdown += "\n"
        
        return markdown.strip()
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def save_config(self, output_path: str = None) -> None:
        """Save configuration to YAML file"""
        path = output_path or self.config_path
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def get_formatted_paths(self, date: str, run_number: str) -> Dict[str, str]:
        """Get formatted paths with date and run number"""
        output_dir = self.config["output"]["dir"].format(date=date, run_number=run_number)
        tracking_cfg = self.config.get("tracking", {})
        tracking_run_tmpl = tracking_cfg.get("run_name_template")
        if isinstance(tracking_run_tmpl, str):
            run_name = tracking_run_tmpl.format(date=date, run_number=run_number)
        else:
            run_name = self.config["output"]["run_name_template"].format(date=date, run_number=run_number)
        
        return {
            "output_dir": output_dir,
            "run_name": run_name
        }

    def get_formatted_hf_repo_id(self, date: str, run_number: str) -> str | None:
        """Return formatted huggingface repo_id if present (supports {date}, {run_number})."""
        hf_cfg = self.config.get("huggingface", {})
        repo_id = hf_cfg.get("repo_id")
        if isinstance(repo_id, str):
            try:
                return repo_id.format(date=date, run_number=run_number)
            except Exception:
                return repo_id
        return None
