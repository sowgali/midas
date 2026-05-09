import type {
  DealDetailDto,
  DealDto,
  EntityDto,
  GraphResponse,
} from "./types";

// All requests go through the Vite dev proxy (or a same-origin reverse proxy
// in prod) at /api, so we never need to think about CORS.
const API_BASE = "/api";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      `API ${res.status} ${res.statusText} for ${path}${body ? `: ${body}` : ""}`,
    );
  }
  return (await res.json()) as T;
}

function qs(params: Record<string, string | undefined>): string {
  const entries = Object.entries(params).filter(
    (entry): entry is [string, string] =>
      entry[1] !== undefined && entry[1] !== "",
  );
  if (entries.length === 0) return "";
  const search = new URLSearchParams(entries);
  return `?${search.toString()}`;
}

export interface GraphFilters {
  sector?: string;
  as_of?: string;
}

export function fetchGraph(filters: GraphFilters): Promise<GraphResponse> {
  return getJson<GraphResponse>(
    `/graph${qs({ sector: filters.sector, as_of: filters.as_of })}`,
  );
}

export function fetchDealsByPair(
  from_id: string,
  to_id: string,
): Promise<DealDto[]> {
  return getJson<DealDto[]>(`/deals${qs({ from_id, to_id })}`);
}

export function fetchDealDetail(deal_id: string): Promise<DealDetailDto> {
  return getJson<DealDetailDto>(`/deals/${encodeURIComponent(deal_id)}`);
}

export function fetchEntities(sector?: string): Promise<EntityDto[]> {
  return getJson<EntityDto[]>(`/entities${qs({ sector })}`);
}
