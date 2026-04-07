# Few-Shot Multi-Task Learning for Universal AI Assistant

Meta-learning agent that adapts to new tasks with minimal training examples using few-shot learning and multi-task optimization.

## Features

- MAML-based meta-learning for rapid task adaptation
- Multi-task training across diverse NLP and vision tasks
- Few-shot evaluation on held-out task distributions
- Task embedding and similarity analysis
- Configurable shot count and task complexity

## Tech Stack

Python, PyTorch, Transformers, NumPy

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/karthik-idikuda/Few-Shot-Multi-Task-Learning-for-Universal-AI-Assistant.git
cd Few-Shot-Multi-Task-Learning-for-Universal-AI-Assistant
pip install -r requirements.txt
```

### Usage

```bash
python train.py --shots 5 --tasks 10
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
