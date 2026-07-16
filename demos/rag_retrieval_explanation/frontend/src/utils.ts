export function formatScore(value: number, signed = true): string {
  if (!signed) return value.toFixed(3);
  return (value >= 0 ? "+" : "") + value.toFixed(3);
}

export function truncateText(text: string, maxChars = 360): string {
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars).trimEnd() + "…";
}

export function buildChunkPreview(text: string): string {
  return truncateText(text, 380);
}
