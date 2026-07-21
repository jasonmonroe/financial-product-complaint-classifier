# Financial Product Complaint Classifier

![Hero Image](data/hero.png)

Classifies customer complaints of financial products

## Description
In today’s financial landscape, customer complaints offer critical insight into service gaps and operational
inefficiencies. Automatically categorizing these complaints into product-specific segments—such as credit reporting,
student loans, or money transfers—helps organizations streamline case handling and direct issues to the appropriate
teams. By applying Generative AI for text classification and summarization, institutions can better interpret customer
concerns and accelerate response workflows. Additionally, generating concise summaries of long complaints enables
support teams to quickly understand the core issue without manually parsing lengthy narratives.

## Objective
This project demonstrates how Generative AI techniques can be applied to enhance both the classification and
summarization of financial customer complaints. Specifically, it focuses on:

Text-to-Label Classification: Using Zero-shot and Few-shot prompting strategies to assign customer complaints to their
correct product categories without requiring traditional supervised training datasets.
Text-to-Text Summarization: Applying Zero-shot prompting to produce clear, concise summaries that help support teams
rapidly interpret customer issues.

## Conclusion
By completing this project, you will gain hands-on experience developing LLM-driven solutions for text classification
and summarization. These capabilities enable financial institutions to automate key aspects of the complaint triage
process—resulting in faster routing, more accurate responses, improved customer satisfaction, and enhanced regulatory
compliance. The techniques demonstrated here also provide transferable skills applicable across a broad range of
real-world NLP and enterprise automation scenarios.

## Purpose
The project is designed to address a common bottleneck in financial services: the manual triaging of customer complaints.
By using Generative AI (specifically the Mistral 7B model via llama-cpp), the system attempts to automatically
categorize complaints into specific product segments and generate concise summaries of the narratives.
Business Objective: The core objective is operational efficiency. By automating classification (Text-to-Label) and
summarization (Text-to-Text), an organization can:

• Reduce Lead Time: Issues reach the correct specialized team (e.g., the Mortgage team vs. the Credit Card team)
immediately.
• Improve Triage Accuracy: Using Few-shot prompting to guide the LLM ensures that classifications align with historical
business labels without needing to retrain a heavy model.
• Standardize Summaries: It helps support agents grasp the "core issue" of a lengthy complaint in seconds, rather than
minutes.

## Evaluation
• Zero-Shot Classification: The code defines a system_message and zero_shot_prompt_template, then uses
get_zero_shot_mistral_response to classify data. It then validates this against the actual product labels using f1_score.

• Few-Shot Classification: The code uses create_examples_df to build a "gold" set of examples (balanced across classes)
and injects them into the prompt context using create_few_shot_prompt. This directly mirrors the objective of improving
performance via examples.

• Summarization: The pipeline includes a dedicated section for summarization, utilizing a different system_message
focused on key points (problem, company, outcome). It uses BERTScore for evaluation, which is a modern, high-standard
metric for text-to-text generation.

• Execution Flow: The argparse implementation allows for modular execution (running only the EDA or the full pipeline),
which is standard for enterprise-level CLI tools.

• Zero-Shot Classification: The code defines a system_message and zero_shot_prompt_template, then uses
get_zero_shot_mistral_response to classify data. It then validates this against the actual product labels using f1_score.

• Few-Shot Classification: The code uses create_examples_df to build a "gold" set of examples (balanced across classes)
and injects them into the prompt context using create_few_shot_prompt. This directly mirrors the objective of improving
performance via examples.

• Summarization: The pipeline includes a dedicated section for summarization, utilizing a different system_message
focused on key points (problem, company, outcome). It uses BERTScore for evaluation, which is a modern, high-standard
metric for text-to-text generation.

• Execution Flow: The argparse implementation allows for modular execution (running only the EDA or the full pipeline),
which is standard for enterprise-level CLI tools.

## Project Structure

```text
customer-review-classifier/
├── main.py                     # CLI entry point for data seeding, EDA, and model execution
├── requirements.txt            # Python dependencies
├── data/                       # Dataset storage
│   ├── hero.png                # Hero graphic / banner image
│   └── source_data.csv         # Standard input dataset
├── src/
│   ├── config.py               # Application configurations and constants
│   ├── eda.py                  # Exploratory Data Analysis module
│   ├── modeling.py             # Model training, prediction, and helper functions
│   ├── preprocessing.py      # Data loading and cleaning pipelines
│   ├── seeder.py               # Synthetic dataset generator module
│   └── utils.py                # Logging, plotting, and common utilities
└── venv/                       # Virtual environment (git-ignored)
```

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd financial-product-complaint-classifier
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
