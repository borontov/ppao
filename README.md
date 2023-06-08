
<div align="center">
    <img src="images/ppao.png" alt="Logo" height="130">

  <h3 align="center">Python Pipeline Algorithmic Optimizer</h3>

  <p align="center">
    Reduce repeated data handler calls by the operations grouping.

  </p>
</div>


## Do I need the ppao?
[![Use cases][use-cases-pic]]()

Even if you cache your heavy handlers in the NoSQL database, you can still reduce the number of queries to the database with ppao.
## What is the idea?
Often, multiple single-threaded workers are used to handle queues of pipelines:

[![Idea][idea-pic]]()

The algorithm allows you to increase the amount of data processed by the handler at a time by grouping operations in the pipelines in such a way that the sequence of the pipelines is not broken.

[![Idea][idea-2-pic]]()

## Benefits:
* You can significantly reduce the number of calls to data handlers or requests to the database to retrieve a cached handler.
* You can group requests to a third-party API to send more data in a single transaction, if that's more advantageous in your case.

## Limitations:
* Algorithm is only suitable for the architecture described in the pictures above.
* The maximum number of operations in the pipelines must be limited and all pipelines must be the same length. If there are fewer operations, zeros are placed in the empty space.
* You should be ready to add the numpy dependency to your project.
* Not all pipelines may be suitable for using this algorithm. The "Grouper" is responsible for checking this. If a pipeline cannot be grouped with others, it will have to wait for new pipelines with which it can form a group to successfully solve the problem. If you don't want to wait, execute the pipeline without ppao. The decision is up to you.


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

[use-cases-pic]: images/use_cases.png
[idea-pic]: images/idea.png
[idea-2-pic]: images/idea_2.png
[solver-pic]: images/solver.png
[grouper-pic]: images/grouper.png
[horizontal-optimizer-pic]: images/horizontal_optimizer.png

