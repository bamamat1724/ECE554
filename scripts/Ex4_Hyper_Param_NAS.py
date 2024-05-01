from pathlib import Path

import torchx

from torchx import specs
from torchx.components import utils


def trainer(
    log_path: str,
    hidden_size: int,
    num_layers: int,
    learning_rate: float,
    epochs: int,
    trial_idx: int = -1,
) -> specs.AppDef:

    # define the log path so we can pass it to the TorchX ``AppDef``
    if trial_idx >= 0:
        log_path = Path(log_path).joinpath(str(trial_idx)).absolute().as_posix()

    return utils.python(
        # command line arguments to the training script
        "--log_path",
        log_path,
        "--hidden_size",
        str(hidden_size),
        "--num_layers",
        str(num_layers),
        "--learning_rate",
        str(learning_rate),
        "--epochs",
        str(epochs),
        # other config options
        name="trainer",
        script=".\\scripts\\Ex2_Train_Kaggle.py",
        image=torchx.version.TORCHX_IMAGE,
    )

import tempfile
from ax.runners.torchx import TorchXRunner

# Make a temporary dir to log our results into
log_dir = tempfile.mkdtemp()

ax_runner = TorchXRunner(
    tracker_base="/tmp/",
    component=trainer,
    # NOTE: To launch this job on a cluster instead of locally you can
    # specify a different scheduler and adjust arguments appropriately.
    scheduler="local_cwd",
    component_const_params={"log_path": log_dir},
    cfg={},
)

from ax.core import (
    ChoiceParameter,
    ParameterType,
    RangeParameter,
    SearchSpace,
)

parameters = [
    # NOTE: In a real-world setting, hidden_size and num_layers
    # should probably be powers of 2, but in our simple example this
    # would mean that ``num_params`` can't take on that many values, which
    # in turn makes the Pareto frontier look pretty weird.
    RangeParameter(
        name="hidden_size",
        lower=16,
        upper=128,
        parameter_type=ParameterType.INT,
        log_scale=True,
    ),
    RangeParameter(
        name="num_layers",
        lower=1,
        upper=5,
        parameter_type=ParameterType.INT,
        log_scale=True,
    ),
    RangeParameter(
        name="learning_rate",
        lower=1e-4,
        upper=1e-2,
        parameter_type=ParameterType.FLOAT,
        log_scale=True,
    ),
    RangeParameter(
        name="epochs",
        lower=10,
        upper=20,
        parameter_type=ParameterType.INT,
    ),
]

search_space = SearchSpace(
    parameters=parameters,
    # NOTE: In practice, it may make sense to add a constraint
    # num_layers <= hidden_size
    parameter_constraints=[],
)

from ax.metrics.tensorboard import TensorboardCurveMetric


class MyTensorboardMetric(TensorboardCurveMetric):

    # NOTE: We need to tell the new TensorBoard metric how to get the id /
    # file handle for the TensorBoard logs from a trial. In this case
    # our convention is to just save a separate file per trial in
    # the prespecified log dir.
    @classmethod
    def get_ids_from_trials(cls, trials):
        return {
            trial.index: Path(log_dir).joinpath(str(trial.index)).as_posix()
            for trial in trials
        }

    # This indicates whether the metric is queryable while the trial is
    # still running. We don't use this in the current tutorial, but Ax
    # utilizes this to implement trial-level early-stopping functionality.
    @classmethod
    def is_available_while_running(cls):
        return False

val_acc = MyTensorboardMetric(
    name="val_acc",
    curve_name="val_acc",
    lower_is_better=False,
)
model_num_params = MyTensorboardMetric(
    name="num_params",
    curve_name="num_params",
    lower_is_better=True,
)

from ax.core import MultiObjective, Objective, ObjectiveThreshold
from ax.core.optimization_config import MultiObjectiveOptimizationConfig


opt_config = MultiObjectiveOptimizationConfig(
    objective=MultiObjective(
        objectives=[
            Objective(metric=val_acc, minimize=False),
            Objective(metric=model_num_params, minimize=True),
        ],
    ),
    objective_thresholds=[
        ObjectiveThreshold(metric=val_acc, bound=0.94, relative=False),
        ObjectiveThreshold(metric=model_num_params, bound=80_000, relative=False),
    ],
)

from ax.core import Experiment

experiment = Experiment(
    name="torchx_lstm",
    search_space=search_space,
    optimization_config=opt_config,
    runner=ax_runner,
)

total_trials = 48  # total evaluation budget

from ax.modelbridge.dispatch_utils import choose_generation_strategy

gs = choose_generation_strategy(
    search_space=experiment.search_space,
    optimization_config=experiment.optimization_config,
    num_trials=total_trials,
  )

from ax.service.scheduler import Scheduler, SchedulerOptions

scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=gs,
    options=SchedulerOptions(
        total_trials=total_trials, max_pending_trials=4
    ),
)

scheduler.run_all_trials()

from ax.service.utils.report_utils import exp_to_df

df = exp_to_df(experiment)
df.head(10)

from ax.service.utils.report_utils import _pareto_frontier_scatter_2d_plotly

_pareto_frontier_scatter_2d_plotly(experiment)

from ax.modelbridge.cross_validation import compute_diagnostics, cross_validate
from ax.plot.diagnostic import interact_cross_validation_plotly
from ax.utils.notebook.plotting import init_notebook_plotting, render

cv = cross_validate(model=gs.model)  # The surrogate model is stored on the ``GenerationStrategy``
compute_diagnostics(cv)

interact_cross_validation_plotly(cv)

from ax.plot.contour import interact_contour_plotly

interact_contour_plotly(model=gs.model, metric_name="val_acc")

interact_contour_plotly(model=gs.model, metric_name="num_params")