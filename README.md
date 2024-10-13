
<div align="center">
    <img src="images/ppao.png" alt="Logo" height="130">

  <h3 align="center">Pipeline Algorithmic Optimizer</h3>

  <p align="center">
    Group data to reduce overhead
  </p>
</div>


## What problem does it solve?
You might find this useful if you have pipelines that use bulk data processors such as:
- NLP tools
- Neural network models
- Third-party APIs
- etc.

And you want to reduce overhead, such as:
- redundant initializations of handler classes
- latency from redundant calls to third-party APIs
- redundant bulk function calls



## Benefits:
For data processing:
* You can optimize huge queues of pipelines by reducing the number of calls for certain operations.

For serverless architecture:
* Given the cold start problem, you can increase the amount of data that will be processed by the lambda function in a single call.

For web services:
* You can group the data sent to a third-party API, if that's more advantageous in your case.

For microservices that process job queues:
* You can aggregate the pipelines that can be optimized and skip the rest as it is.

## Limitations:
* Algorithm is only suitable for the bulk functions pipelines.
* The maximum number of operations in the pipelines must be limited and all pipelines must be the same length. If there are fewer operations, zeros are placed in the empty space. You can also use the chain design pattern.
* You should be ready to add the numpy dependency to your project.
* Not all sets of pipelines may be suitable for using this algorithm. The Grouper is responsible for checking this. If a pipeline cannot be grouped with others, it will have to wait for new pipelines with which it can form a group to successfully solve the problem. You can control what is in the grouper and execute the pipelines yourself when you need to.

## What is the idea?
The pipeline is a pattern used to process data or tasks in a series of sequential steps. Data passes through a series of handlers, where each step performs its specific function and passes the result to the next step.

Note: The data in the pictures below should not be taken as an object but as a reference to it, because the data in the pipelines is usually transformed at each step.

[![Idea][idea-pic]]()

This algorithm solves the problem of merging several pipelines into one by combining the same operations. The order of operations for each participating pipelines is not affected.

Bulk functions in your pipelines that have the same arguments, except for the data passed for processing, are considered equivalent operations that can be merged into one.

[![Idea][idea-2-pic]]()

An attentive eye will notice that in this illustration there are many equivalent operations in one pipeline, which is rarely the case in practice. This is a good notice, but actually the lack of identical operations in the pipline doesn't affect the efficiency of the algorithm. If there are too many colors, it will be harder to understand the illustrations, so don't pay too much attention to that.

You may have noticed that there are significantly fewer operations. Magic? Nope. Next, I will explain how it works. I've hidden the description of each component under the clickable dropdown.

## Explanation of the algorithm.
### Components:
<details>
  <summary>Solver</summary>

[![Solver][solver-pic]]()
</details>

<details>
  <summary>Grouper</summary>

[![Grouper][grouper-pic]]()
</details>

<details>
  <summary>Horizontal optimizer</summary>

[![Horizontal Optimizer][horizontal-optimizer-pic]]()
</details>

## Dependencies:
* numpy
* python >= 3.10

## Installation:

### With pip:
   ```sh
   pip install ppao
   ```

### Git:
   ```sh
   git clone https://github.com/borontov/ppao.git
   ```

### poetry:

   ```sh
   poetry install
   ```

### conda-lock:

   ```sh
   conda-lock install --micromamba -n ppao_dev_env conda-lock.yml
   ```

## Getting Started

### Example:

  ```python
from ppao import (
    Grouper,
    PipelineMatrixSolver,
    settings as ppao_settings,
)
import numpy as np

settings_ = ppao_settings.Settings(
    group_size_limit=4,
    pipeline_size_limit=4,
    common_ops_percent_bound=0.85,
)
pipelines = np.array(
    [
        [1, 3, 1, 2],
        [1, 1, 1, 2],
        [3, 2, 1, 1],
        [1, 2, 2, 1],
    ],
)
grouper = Grouper(settings_=settings_)
grouper.add(pipelines=pipelines)
source_matrix = grouper.pop()
solver = PipelineMatrixSolver(
    source_matrix=source_matrix,
    settings_=settings_,
)
solution = solver.solve()
print(solution)

# output:
# [
#   ExecutionUnit(operation=1, pipelines=array([0, 2, 0, 1], dtype=uint16)),
#   ExecutionUnit(operation=3, pipelines=array([2, 3], dtype=uint16)),
#   ExecutionUnit(operation=1, pipelines=array([0, 2], dtype=uint16)),
#   ExecutionUnit(operation=2, pipelines=array([1, 3, 0, 1, 2], dtype=uint16)),
#   ExecutionUnit(operation=1, pipelines=array([3, 1, 3], dtype=uint16))
# ]
  ```

### Solution usage:

```python
for execution_unit in solution:
    handler = get_handler(execution_unit.operation)
    for pipeline_id in execution_unit.pipelines:
        pipeline_data = get_pipeline_data(pipeline_id)
        handler(pipeline_data)
```


## Roadmap

- [ ] Add debug logging
- [ ] Deduplicate equal shift combinations like (-1, -1, 0, 0) & (0, 0, 1, 1)
- [ ] PyPI-friendly README

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).





## Contributing

Contributions are welcome.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Prepare your feature with shell commands:
   * make format
   * make check
   * make test
4. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the Branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

[idea-pic]: images/idea.png
[idea-2-pic]: images/idea_2.png
[solver-pic]: images/solver.png
[grouper-pic]: images/grouper.png
[horizontal-optimizer-pic]: images/horizontal_optimizer.png

