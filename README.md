# LLM Evaluation of Medical Videos

This folder contains code and scripts for using large language models (LLMs) in evluating the credibility of YouTube medical videos. Evaluation is done on the transcripts of the videos, which was generated using [this script](https://github.com/mm-khalil1/YouTube-Transcript-Generator).

## Subfolders

- **Claude_Evaluation**

- **Gemini_Evaluation**

- **HuggingFace_Evaluation**

- **OpenAI_Evaluation**

- **Results_Analysis**

  Contains scripts for analyzing the results of LLM evaluations:

  - **all_LLMs_plots_and_statistics.ipynb**

    - Script for generating plots and statistics for evaluations of all LLMs.

  - **analyse_LLMs_responses.ipynb**

    - Script for analyzing the responses of each LLM.

  - **one_LLM_plots_and_statistics.ipynb**

    - Script for generating plots and statistics for a single LLM.

  - **statistics_plots_analysis_utils.py**

    - Utility functions for generating plots and analyzing statistics.

## Files

- **llm_evaluation_utils.py**
  - Utility functions for LLM evaluation.
