# Data

`mbpp_test.jsonl` (500 tasks, ids 11-510) and `mbpp_prompt.jsonl` (10 tasks) are the
original MBPP splits (Austin et al., 2021, "Program Synthesis with Large Language
Models"), from google-research/mbpp, licensed CC BY 4.0. Fields used: `task_id`, `text`,
`test_list`, `test_setup_code`; `code` (reference solution) is only used by the scripted
`smoke` check, never shown to the evaluated models.
