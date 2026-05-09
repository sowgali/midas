import type { JSX } from "react";

import type { GraphFilters } from "../api";
import styles from "./FilterPanel.module.css";

const SECTOR_OPTIONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "", label: "All sectors" },
  { value: "ai", label: "AI" },
  { value: "cloud", label: "Cloud" },
  { value: "semiconductors", label: "Semiconductors" },
  { value: "software", label: "Software" },
];

interface FilterPanelProps {
  filters: GraphFilters;
  onChange: (next: GraphFilters) => void;
}

export default function FilterPanel({
  filters,
  onChange,
}: FilterPanelProps): JSX.Element {
  return (
    <div className={styles.panel}>
      <h2 className={styles.heading}>Filters</h2>

      <label className={styles.field}>
        <span className={styles.label}>Sector</span>
        <select
          className={styles.control}
          value={filters.sector ?? ""}
          onChange={(e) =>
            onChange({ ...filters, sector: e.target.value || undefined })
          }
        >
          {SECTOR_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </label>

      <label className={styles.field}>
        <span className={styles.label}>As of</span>
        <input
          type="date"
          className={styles.control}
          value={filters.as_of ?? ""}
          onChange={(e) =>
            onChange({ ...filters, as_of: e.target.value || undefined })
          }
        />
      </label>

      <p className={styles.hint}>
        Click a node or edge to inspect deals and provenance.
      </p>
    </div>
  );
}
