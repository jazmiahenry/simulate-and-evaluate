import os
import subprocess
import argparse
from pathlib import Path

def setup_environment(model_type: str):
    """Setup training environment"""
    # Create .env file
    env_content = f"""
    MODEL_TYPE={model_type}
    WANDB_PROJECT=llm-behavior-comparison
    DATA_DIR=./data
    OUTPUT_DIR=./outputs
    """
    
    if model_type == "gpt4":
        env_content += """
        # Add your OpenAI API key here
        OPENAI_API_KEY=your-key-here
        """
    
    with open(".env", "w") as f:
        f.write(env_content.strip())
    
    # Create directory structure
    directories = ["data", "outputs", "logs", "configs"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Initialize git repository if not already initialized
    if not Path(".git").exists():
        subprocess.run(["git", "init"])
        
        # Create .gitignore
        gitignore_content = """
        .env
        __pycache__/
        *.pyc
        outputs/
        logs/
        wandb/
        *.pt
        *.pth
        """
        with open(".gitignore", "w") as f:
            f.write(gitignore_content.strip())

def main():
    parser = argparse.ArgumentParser(description="Setup LLM training environment")
    parser.add_argument(
        "--model-type",
        choices=["llama", "gpt4"],
        required=True,
        help="Type of model to setup"
    )
    
    args = parser.parse_args()
    setup_environment(args.model_type)
    
    print("""
    Environment setup complete! Next steps:
    
    1. If using GPT-4:
       - Add your OpenAI API key to the .env file
       
    2. If using LLaMA:
       - Ensure you have access to the LLaMA model weights
       - Update LLAMA_MODEL_PATH in .env if needed
       
    3. Install dependencies:
       pip install -r requirements.txt
       
    4. Login to Weights & Biases:
       wandb login
       
    5. Start training:
       python train.py
    """)

if __name__ == "__main__":
    main()