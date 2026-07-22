"""Public execution worker facade.

The implementation is split across execution_* mixins to keep each source file
small while preserving the historical ``planning.execution.PlanningWorker``
import path used by desktop entrypoints.
"""

from .execution_common import *
from .execution_base import ExecutionBaseMixin
from .execution_plan_state import ExecutionPlanStateMixin
from .execution_setup_online import ExecutionSetupOnlineMixin
from .execution_step import ExecutionStepMixin
from .execution_perception import ExecutionPerceptionMixin
from .execution_manual import ExecutionManualMixin
from .execution_scene_scan import ExecutionSceneScanMixin
from .execution_motion_segments import ExecutionMotionSegmentsMixin
from .execution_diagnostics import ExecutionDiagnosticsMixin
from .execution_viewer import ExecutionViewerMixin


class PlanningWorker(
    ExecutionBaseMixin,
    ExecutionPlanStateMixin,
    ExecutionSetupOnlineMixin,
    ExecutionStepMixin,
    ExecutionPerceptionMixin,
    ExecutionManualMixin,
    ExecutionSceneScanMixin,
    ExecutionMotionSegmentsMixin,
    ExecutionDiagnosticsMixin,
    ExecutionViewerMixin,
):
    """Field execution worker used by desktop GUI entrypoints."""

    pass
