import { describe, it, expect } from "vitest";
import { sleep, assert } from "./util";

describe("sleep", () => {
  it("should resolve after specified time", async () => {
    const start = Date.now();
    await sleep(100);
    const elapsed = Date.now() - start;
    expect(elapsed).toBeGreaterThanOrEqual(95); // Allow for small timing variations
  });

  it("should resolve immediately for 0ms", async () => {
    const start = Date.now();
    await sleep(0);
    const elapsed = Date.now() - start;
    expect(elapsed).toBeLessThan(50);
  });
});

describe("assert", () => {
  it("should not throw for truthy condition", () => {
    expect(() => {
      assert(true);
    }).not.toThrow();
    expect(() => {
      assert(1);
    }).not.toThrow();
    expect(() => {
      assert("hello");
    }).not.toThrow();
    expect(() => {
      assert({});
    }).not.toThrow();
  });

  it("should throw for falsy condition", () => {
    expect(() => {
      assert(false);
    }).toThrow();
    expect(() => {
      assert(0);
    }).toThrow();
    expect(() => {
      assert("");
    }).toThrow();
    expect(() => {
      assert(null);
    }).toThrow();
    expect(() => {
      assert(undefined);
    }).toThrow();
  });

  it("should throw with custom message", () => {
    expect(() => {
      assert(false, "Custom error");
    }).toThrow("Custom error");
  });

  it("should throw with default message", () => {
    expect(() => {
      assert(false);
    }).toThrow("Assertion failed");
  });
});
