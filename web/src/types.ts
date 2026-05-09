// TypeScript mirrors of the FastAPI contracts. Keep these in sync with
// the Pydantic DTOs defined on the backend.

export interface EntityDto {
  id: string;
  canonical_name: string;
  aliases: string[];
  ticker: string | null;
  cik: string | null;
  entity_type: string;
  sector_tags: string[];
  country: string | null;
}

export interface SourceDto {
  id: string;
  url: string;
  source_type: string;
  publisher: string;
  title: string | null;
  published_at: string | null; // ISO datetime
}

export interface EvidenceDto {
  id: string;
  text_snippet: string;
  char_start: number;
  char_end: number;
  extractor: string;
  source: SourceDto;
}

export interface DealDto {
  id: string;
  from_entity_id: string;
  to_entity_id: string;
  deal_type: string;
  status: string;
  amount_usd: number | null;
  amount_native: number | null;
  currency: string | null;
  announced_at: string | null; // ISO date
  closes_at: string | null;
  confidence: number;
  description: string;
}

export interface DealDetailDto extends DealDto {
  from_entity: EntityDto;
  to_entity: EntityDto;
  evidence: EvidenceDto[];
}

export interface GraphEdgeDto {
  from_id: string;
  to_id: string;
  total_amount_usd: number | null;
  deal_count: number;
  deal_types: string[];
}

export interface GraphResponse {
  nodes: EntityDto[];
  edges: GraphEdgeDto[];
  as_of: string | null;
  sector: string | null;
}
