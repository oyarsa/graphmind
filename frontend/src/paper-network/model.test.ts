import { describe, it, expect } from "vitest";
import { getNodeId, getNodePos, D3Node } from "./model";

describe("getNodeId", () => {
  it("should return string as-is", () => {
    expect(getNodeId("paper-123")).toBe("paper-123");
  });

  it("should convert number to string", () => {
    expect(getNodeId(42)).toBe("42");
    expect(getNodeId(0)).toBe("0");
  });

  it("should extract id from D3Node object", () => {
    const node: D3Node = {
      id: "node-456",
      title: "Test Paper",
      authors: ["Author 1"],
      year: 2023,
      abstract: "Test abstract",
      venue: "Test Venue",
      citation_count: 10,
    };

    expect(getNodeId(node)).toBe("node-456");
  });

  it("should handle D3Node with position data", () => {
    const node: D3Node = {
      id: "positioned-node",
      title: "Positioned Paper",
      authors: ["Author 1"],
      year: 2023,
      abstract: "Test abstract",
      venue: "Test Venue",
      citation_count: 5,
      x: 100,
      y: 200,
    };

    expect(getNodeId(node)).toBe("positioned-node");
  });
});

describe("getNodePos", () => {
  it("should return empty object for string", () => {
    expect(getNodePos("paper-123")).toEqual({});
  });

  it("should return empty object for number", () => {
    expect(getNodePos(42)).toEqual({});
  });

  it("should extract position from D3Node without position", () => {
    const node: D3Node = {
      id: "node-456",
      title: "Test Paper",
      authors: ["Author 1"],
      year: 2023,
      abstract: "Test abstract",
      venue: "Test Venue",
      citation_count: 10,
    };

    const result = getNodePos(node);
    expect(result).toEqual({ x: undefined, y: undefined });
  });

  it("should extract position from D3Node with position", () => {
    const node: D3Node = {
      id: "positioned-node",
      title: "Positioned Paper",
      authors: ["Author 1"],
      year: 2023,
      abstract: "Test abstract",
      venue: "Test Venue",
      citation_count: 5,
      x: 150,
      y: 250,
    };

    const result = getNodePos(node);
    expect(result).toEqual({ x: 150, y: 250 });
  });

  it("should handle partial position data", () => {
    const nodeXOnly: D3Node = {
      id: "x-only-node",
      title: "X Only Paper",
      authors: ["Author 1"],
      year: 2023,
      abstract: "Test abstract",
      venue: "Test Venue",
      citation_count: 3,
      x: 100,
    };

    const resultXOnly = getNodePos(nodeXOnly);
    expect(resultXOnly).toEqual({ x: 100, y: undefined });

    const nodeYOnly: D3Node = {
      id: "y-only-node",
      title: "Y Only Paper",
      authors: ["Author 1"],
      year: 2023,
      abstract: "Test abstract",
      venue: "Test Venue",
      citation_count: 3,
      y: 200,
    };

    const resultYOnly = getNodePos(nodeYOnly);
    expect(resultYOnly).toEqual({ x: undefined, y: 200 });
  });

  it("should handle zero coordinates", () => {
    const node: D3Node = {
      id: "origin-node",
      title: "Origin Paper",
      authors: ["Author 1"],
      year: 2023,
      abstract: "Test abstract",
      venue: "Test Venue",
      citation_count: 1,
      x: 0,
      y: 0,
    };

    const result = getNodePos(node);
    expect(result).toEqual({ x: 0, y: 0 });
  });
});
