import { useState, type JSX } from "react";
import { useQuery } from "@tanstack/react-query";

import { fetchDealDetail, fetchDealsByPair } from "../api";
import { entityTypeColor, formatUsd } from "../format";
import type { DealDetailDto, DealDto, EntityDto } from "../types";
import type { Selection } from "../App";
import styles from "./DetailDrawer.module.css";

interface DetailDrawerProps {
  selection: Selection;
  onClose: () => void;
}

export default function DetailDrawer({
  selection,
  onClose,
}: DetailDrawerProps): JSX.Element {
  if (!selection) {
    return (
      <div className={styles.empty}>
        <p>Select a node or edge to see details.</p>
      </div>
    );
  }

  return (
    <div className={styles.drawer}>
      <header className={styles.header}>
        <h2 className={styles.heading}>
          {selection.kind === "node" ? "Entity" : "Deals"}
        </h2>
        <button
          type="button"
          className={styles.closeBtn}
          onClick={onClose}
          aria-label="Close detail drawer"
        >
          ×
        </button>
      </header>

      {selection.kind === "node" ? (
        <EntityDetail entity={selection.entity} />
      ) : (
        <PairDeals from={selection.from} to={selection.to} />
      )}
    </div>
  );
}

function EntityDetail({ entity }: { entity: EntityDto }): JSX.Element {
  const color = entityTypeColor(entity.entity_type);
  return (
    <section className={styles.section}>
      <div className={styles.entityHeader}>
        <span
          className={styles.bigDot}
          style={{ backgroundColor: color }}
          aria-hidden="true"
        />
        <div>
          <h3 className={styles.entityName}>{entity.canonical_name}</h3>
          {entity.ticker ? (
            <span className={styles.ticker}>{entity.ticker}</span>
          ) : null}
        </div>
      </div>

      <Field label="Type" value={entity.entity_type} />
      {entity.country ? <Field label="Country" value={entity.country} /> : null}
      {entity.cik ? <Field label="CIK" value={entity.cik} /> : null}

      {entity.sector_tags.length > 0 ? (
        <div className={styles.field}>
          <span className={styles.fieldLabel}>Sector tags</span>
          <div className={styles.chips}>
            {entity.sector_tags.map((tag) => (
              <span key={tag} className={styles.chip}>
                {tag}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {entity.aliases.length > 0 ? (
        <div className={styles.field}>
          <span className={styles.fieldLabel}>Aliases</span>
          <div className={styles.chips}>
            {entity.aliases.map((alias) => (
              <span key={alias} className={styles.chip}>
                {alias}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function Field({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div className={styles.field}>
      <span className={styles.fieldLabel}>{label}</span>
      <span className={styles.fieldValue}>{value}</span>
    </div>
  );
}

function PairDeals({
  from,
  to,
}: {
  from: EntityDto;
  to: EntityDto;
}): JSX.Element {
  const { data, isLoading, isError, error } = useQuery<DealDto[], Error>({
    queryKey: ["deals", from.id, to.id],
    queryFn: () => fetchDealsByPair(from.id, to.id),
  });

  return (
    <section className={styles.section}>
      <div className={styles.pairHeader}>
        <span className={styles.pairName}>{from.canonical_name}</span>
        <span className={styles.pairArrow}>→</span>
        <span className={styles.pairName}>{to.canonical_name}</span>
      </div>

      {isLoading ? (
        <p className={styles.muted}>Loading deals…</p>
      ) : isError ? (
        <p className={styles.error}>Failed to load deals: {error.message}</p>
      ) : !data || data.length === 0 ? (
        <p className={styles.muted}>No deals recorded between this pair.</p>
      ) : (
        <ul className={styles.dealList}>
          {data.map((deal) => (
            <DealCard key={deal.id} deal={deal} />
          ))}
        </ul>
      )}
    </section>
  );
}

function DealCard({ deal }: { deal: DealDto }): JSX.Element {
  const [expanded, setExpanded] = useState(false);

  const { data: detail, isFetching } = useQuery<DealDetailDto, Error>({
    queryKey: ["deal", deal.id],
    queryFn: () => fetchDealDetail(deal.id),
    enabled: expanded,
  });

  const confidencePct = Math.round(deal.confidence * 100);

  return (
    <li className={styles.dealCard}>
      <div className={styles.dealRow}>
        <span className={styles.dealType}>{deal.deal_type}</span>
        <span className={styles.dealAmount}>{formatUsd(deal.amount_usd)}</span>
      </div>
      <div className={styles.dealMetaRow}>
        <span className={styles.dealMeta}>
          {deal.announced_at ?? "date unknown"}
        </span>
        <span className={styles.dealMeta}>{deal.status}</span>
      </div>
      <div className={styles.confidence} title={`Confidence ${confidencePct}%`}>
        <div
          className={styles.confidenceFill}
          style={{ width: `${confidencePct}%` }}
        />
        <span className={styles.confidenceLabel}>{confidencePct}%</span>
      </div>
      {deal.description ? (
        <p className={styles.dealDescription}>{deal.description}</p>
      ) : null}

      <button
        type="button"
        className={styles.expandBtn}
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
      >
        {expanded ? "Hide evidence" : "Show evidence"}
      </button>

      {expanded ? (
        isFetching && !detail ? (
          <p className={styles.muted}>Loading evidence…</p>
        ) : detail ? (
          <Evidence detail={detail} />
        ) : null
      ) : null}
    </li>
  );
}

function Evidence({ detail }: { detail: DealDetailDto }): JSX.Element {
  if (detail.evidence.length === 0) {
    return <p className={styles.muted}>No evidence attached.</p>;
  }
  return (
    <ul className={styles.evidenceList}>
      {detail.evidence.map((ev) => (
        <li key={ev.id} className={styles.evidenceItem}>
          <blockquote className={styles.snippet}>“{ev.text_snippet}”</blockquote>
          <div className={styles.evidenceMeta}>
            <a
              href={ev.source.url}
              target="_blank"
              rel="noreferrer noopener"
              className={styles.sourceLink}
            >
              {ev.source.title ?? ev.source.publisher}
            </a>
            <span className={styles.evidenceMetaSub}>
              {ev.source.publisher}
              {ev.source.published_at
                ? ` · ${new Date(ev.source.published_at).toLocaleDateString()}`
                : ""}
              {" · "}
              {ev.extractor}
            </span>
          </div>
        </li>
      ))}
    </ul>
  );
}
