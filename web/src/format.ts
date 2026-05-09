// Compact USD formatter shared by GraphView and DetailDrawer.
export function formatUsd(amount: number | null | undefined): string {
  if (amount === null || amount === undefined) return "—";
  const abs = Math.abs(amount);
  if (abs >= 1e12) return `$${(amount / 1e12).toFixed(1)}T`;
  if (abs >= 1e9) return `$${(amount / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `$${(amount / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `$${(amount / 1e3).toFixed(1)}K`;
  return `$${amount.toFixed(0)}`;
}

const ENTITY_TYPE_COLORS: Record<string, string> = {
  public_company: "#2563eb", // blue
  private_company: "#ea580c", // orange
  government: "#6b7280", // grey
  fund: "#16a34a", // green
  nonprofit: "#9333ea", // purple
};

export function entityTypeColor(entity_type: string): string {
  return ENTITY_TYPE_COLORS[entity_type] ?? "#94a3b8";
}
