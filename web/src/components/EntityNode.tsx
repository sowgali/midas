import type { JSX } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

import { entityTypeColor } from "../format";
import type { EntityDto } from "../types";
import styles from "./EntityNode.module.css";

export interface EntityNodeData extends Record<string, unknown> {
  entity: EntityDto;
}

export type EntityFlowNode = Node<EntityNodeData, "entity">;

export default function EntityNode({
  data,
}: NodeProps<EntityFlowNode>): JSX.Element {
  const entity = data.entity;
  const color = entityTypeColor(entity.entity_type);
  return (
    <div className={styles.node}>
      <Handle type="target" position={Position.Left} className={styles.handle} />
      <span
        className={styles.dot}
        style={{ backgroundColor: color }}
        title={entity.entity_type}
      />
      <span className={styles.name}>{entity.canonical_name}</span>
      {entity.ticker ? (
        <span className={styles.ticker}>{entity.ticker}</span>
      ) : null}
      <Handle type="source" position={Position.Right} className={styles.handle} />
    </div>
  );
}
