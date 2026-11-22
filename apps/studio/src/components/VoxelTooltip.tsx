import { inspectVoxel } from "../lib/abmesh";
import { useInspectorStore } from "../store/useInspectorStore";

export function VoxelTooltip() {
  const frame = useInspectorStore((state) => state.frame);
  const hovered = useInspectorStore((state) => state.hoveredVoxelId);
  if (!frame || hovered === null) return null;
  const voxel = inspectVoxel(frame, hovered);
  if (!voxel) return null;
  const displacement = voxel.disp_norm * frame.dispP99;
  return (
    <div className="voxel-tooltip">
      Voxel ({voxel.x}, {voxel.y}, {voxel.z}) · ρ={voxel.density.toFixed(2)} · P_add=
      {voxel.policy_add.toFixed(3)} · P_rem={voxel.policy_remove.toFixed(3)} · N=
      {Math.round(voxel.mcts_add + voxel.mcts_remove)} · Q=
      {(voxel.q_add + voxel.q_remove).toFixed(2)} · |u|={displacement.toFixed(1)}
    </div>
  );
}
