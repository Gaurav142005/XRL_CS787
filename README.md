# Explainable Reinforcement Learning with Adversarial LLM Feedback

## Introduction

This repository contains the code and resources for the course project, of the course CS787 : Generative AI, project titled **"Explainable Reinforcement Learning with Adversarial LLM Feedback"**. The project investigates the use of large language models (LLMs) as adversarial agents to improve the explainability and performance of reinforcement learning (RL) systems. It compares the outcomes of using RL alone versus RL assisted with LLM feedback.

We have used LLM using Gemini API, we used Gemini's 2.0-flash model for generating feedback as adversarial and defining the reward while fine tuning RL model.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Gaurav142005/XRL_CS787.git
   cd XRL_CS787
   ```

2. (Optional) Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Setup Environment Variables (Linux):
   ```bash
   echo 'GOOGLE_API_KEY = "your_google_api_key_here"\' > .env
   ```

---

## Usage

### Visualing Our Results

Although we have uploaded our results in the results folder, if you want to visualise the models, then just run:

```bash
python ./codes/visualise.py
```

This is for 6x6 grid.

### Training on device

If you want to train on your device you have to just run:

```bash
python ./codes/training.py
```

And if you want to evaluate and plot comparing graphs then run:

```bash
python ./codes/eval.py
python ./codes/plot_results.py
```
