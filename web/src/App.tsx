import { useState, type JSX } from "react";

import FilterPanel from "./components/FilterPanel";
import GraphView from "./components/GraphView";
import DetailDrawer from "./components/DetailDrawer";
import type { GraphFilters } from "./api";
import type { EntityDto, GraphEdgeDto } from "./types";
import styles from "./App.module.css";

export type Selection =
  | { kind: "node"; entity: EntityDto }
  | { kind: "edge"; edge: GraphEdgeDto; from: EntityDto; to: EntityDto }
  | null;

export default function App(): JSX.Element {
  const [filters, setFilters] = useState<GraphFilters>({
    sector: "",
    as_of: "",
  });
  const [selection, setSelection] = useState<Selection>(null);

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <h1 className={styles.title}>midas</h1>
        <span className={styles.tagline}>AI sector cash flows</span>
      </header>
      <div className={styles.body}>
        <aside className={styles.sidebar}>
          <FilterPanel filters={filters} onChange={setFilters} />
        </aside>
        <main className={styles.canvas}>
          <GraphView filters={filters} onSelect={setSelection} />
        </main>
        <aside className={styles.drawer}>
          <DetailDrawer
            selection={selection}
            onClose={() => setSelection(null)}
          />
        </aside>
      </div>
    </div>
  );
}
