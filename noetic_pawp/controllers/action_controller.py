from __future__ import annotations

from .contracts import ActionOutput, BridgeState, CognitionState


class ActionController:
    def build(self, cognition: CognitionState, bridge_state: BridgeState) -> ActionOutput:
        return ActionOutput(
            prs=bridge_state.prs,
            z_noetic=bridge_state.z_noetic,
            spikes=cognition.spikes,
            membrane=cognition.membrane,
            surprise=cognition.surprise,
            surprise_trace=cognition.surprise_trace,
        )
