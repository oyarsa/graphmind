import { type ZodType } from "zod/v4";

/**
 * Parse JSON from storage using a Zod schema.
 */
export function parseStoredJson<T>(
  raw: string | null,
  schema: ZodType<T>,
  key: string,
): T | null {
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as unknown;
    return schema.parse(parsed);
  } catch (error) {
    console.warn(`[Storage] Invalid JSON for key '${key}':`, error);
    return null;
  }
}
