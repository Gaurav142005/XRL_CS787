# README

## Project: Explainable Reinforcement Learning with LLM Feedback

This repository contains the code and resources for the CS787 project titled **"Explainable Reinforcement Learning with LLM Feedback"**. The project investigates the use of large language models (LLMs) as adversarial agents to enhance the explainability and performance of reinforcement learning (RL) systems. It compares the outcomes of using RL alone versus RL augmented with LLM feedback.

## Introduction

Reinforcement learning systems often operate as black boxes, making their decision-making processes difficult to interpret. This project leverages LLMs as adversarial agents to provide feedback, aiming to improve both the performance and explainability of RL systems. By comparing RL-only systems with RL systems augmented by LLM feedback, the project seeks to highlight the advantages and trade-offs of this hybrid approach.

## Features

- **Explainability Tools**: Methods to interpret and visualize RL decision-making processes.
- **LLM Feedback Integration**: Use LLMs as adversarial agents to challenge and refine RL policies.
- **Performance Comparison**: Analyze the differences between RL-only and RL+LLM systems.
- **Customizable Experiments**: Easily modify parameters to test various scenarios.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/CS787-Explainable_RL_LLM.git
    cd CS787-Explainable_RL_LLM
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) Set up a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  
    ```
---

## Usage

1. Run the main script:
    ```bash
    python visualization.py
    ```

2. View results in the `results/` directory.