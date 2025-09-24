# Few-Shot Multi-Task Learning for Universal AI Assistant

## Overview

This project implements a few-shot multi-task learning framework for building a universal AI assistant capable of rapidly adapting to new tasks with minimal examples. The system combines meta-learning techniques with multi-task learning to create an AI assistant that can handle diverse domains including natural language processing, computer vision, code generation, and reasoning tasks.

## Key Features

- **Few-Shot Learning**: Rapid adaptation to new tasks with minimal training examples
- **Multi-Task Learning**: Shared representations across multiple task domains
- **Meta-Learning**: Model-Agnostic Meta-Learning (MAML) and Prototypical Networks
- **Universal Capabilities**: NLP, Vision, Code Generation, Mathematical Reasoning
- **Modular Architecture**: Easily extensible for new tasks and domains
- **Efficient Training**: Gradient-based meta-learning with task sampling strategies
- **Interactive Interfaces**: Command-line and graphical user interfaces for easy interaction

## Architecture

### Core Components

1. **Meta-Learner**: Implements MAML and other meta-learning algorithms
2. **Multi-Task Backbone**: Shared feature extraction across domains
3. **Task-Specific Heads**: Specialized output layers for different task types
4. **Few-Shot Adapters**: Rapid adaptation mechanisms for new tasks
5. **Universal Interface**: Unified API for different assistant capabilities

### Supported Task Domains

- **Natural Language Processing**: Text classification, generation, summarization
- **Computer Vision**: Image classification, object detection, visual QA
- **Code Generation**: Programming assistance, code completion, debugging
- **Mathematical Reasoning**: Problem solving, equation solving, proof assistance
- **Conversational AI**: Dialog management, context understanding

## Project Structure

```
├── src/
│   ├── core/              # Core meta-learning algorithms
│   ├── models/            # Model architectures
│   ├── tasks/             # Task-specific implementations
│   ├── data/              # Data loading and processing
│   ├── training/          # Training loops and optimization
│   ├── evaluation/        # Metrics and evaluation
│   └── utils/             # Utility functions
├── experiments/           # Experiment configurations
├── data/                 # Dataset storage
├── configs/              # Configuration files
├── notebooks/            # Jupyter notebooks for analysis
├── tests/                # Unit tests
└── requirements.txt      # Dependencies
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd vishnu

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
from src.models.universal_assistant import UniversalAssistant
from src.core.meta_learner import MAMLLearner

# Initialize the universal assistant
assistant = UniversalAssistant(
    backbone="transformer",
    meta_learner="maml",
    task_domains=["nlp", "vision", "code", "math"]
)

# Few-shot adaptation to a new task
support_set = load_support_examples(task="sentiment_analysis", n_shots=5)
assistant.adapt(support_set)

# Use the adapted model
query = "This movie is amazing!"
result = assistant.predict(query, task_type="text_classification")
```

## Usage

### Command Line Interface

```bash
# Interactive mode
python src/cli.py --interactive

# Single-shot mode
python src/cli.py --task sentiment_analysis --input "This movie was fantastic!"
```

### Graphical User Interface (GUI)

```bash
# Make the script executable (one-time setup)
chmod +x run_gui.sh

# Launch the GUI
./run_gui.sh
```

The GUI will automatically open in your default web browser at http://localhost:8501 and provides:
- Interactive task demonstration with few-shot examples
- Task management (create, view, delete tasks)
- Model performance visualization and analytics
- Settings configuration

## Training

```bash
# Train the universal assistant
python scripts/train.py --config configs/universal_assistant.yaml

# Meta-learning on multiple tasks
python scripts/meta_train.py --tasks nlp,vision,code --n_shots 5

# Evaluate few-shot performance
python scripts/evaluate.py --model_path checkpoints/best_model.pt --test_tasks data/test_tasks.json
```

## Experiments

The `experiments/` directory contains various experimental configurations:

- **Baseline Experiments**: Standard multi-task learning without meta-learning
- **MAML Experiments**: Model-Agnostic Meta-Learning implementations
- **Prototypical Networks**: Distance-based few-shot learning
- **Task Sampling Strategies**: Different approaches to task selection during training
- **Cross-Domain Transfer**: Evaluation across different task domains

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

- Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks.
- Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning.
- Caruana, R. (1997). Multitask learning.
- Hospedales, T., et al. (2021). Meta-learning in neural networks: A survey.
# Few-Shot-Multi-Task-Learning-for-Universal-AI-Assistant
