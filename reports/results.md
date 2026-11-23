### Qwen/Qwen2.5-1.5B-Instruct (500 MBPP tasks)

| policy | hidden pass | visible pass | first-try pass | overfit | gens/task | repeat rate | new tokens/task |
|---|---|---|---|---|---|---|---|
| single | 0.428 | 0.502 | 0.428 | 0.147 | 1.00 | 0.000 | 113 |
| resample | 0.534 | 0.652 | 0.428 | 0.181 | 2.29 | 0.005 | 298 |
| repair | 0.458 | 0.544 | 0.418 | 0.158 | 2.46 | 0.575 | 350 |
| repair_t | 0.470 | 0.570 | 0.436 | 0.175 | 2.38 | 0.122 | 377 |
| blind | 0.474 | 0.564 | 0.436 | 0.160 | 2.38 | 0.471 | 310 |

Paired differences in hidden pass rate (95% bootstrap CI):

- repair - single: +0.030 [+0.004, +0.056]
- resample - single: +0.106 [+0.080, +0.132]
- repair_t - single: +0.042 [+0.018, +0.068]
- repair_t - resample: -0.064 [-0.094, -0.034]
- repair - blind: -0.016 [-0.040, +0.008]

First-attempt failure types (single policy):

import: 16, pass: 251, runtime_error: 48, wrong_answer: 185
