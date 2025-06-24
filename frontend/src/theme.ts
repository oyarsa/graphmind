// Theme management utility for dark/light mode toggle
const STORAGE_KEY = "paper-explorer-theme";
const DARK_CLASS = "dark";

/**
 * Apply theme to DOM
 */
function applyTheme(theme: "light" | "dark"): void {
  if (theme === "dark") {
    document.documentElement.classList.add(DARK_CLASS);
  } else {
    document.documentElement.classList.remove(DARK_CLASS);
  }
}

/**
 * Save theme preference to localStorage
 */
function saveTheme(theme: "light" | "dark"): void {
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch (error) {
    console.warn("Failed to save theme preference:", error);
  }
}

/**
 * Get saved theme from localStorage, defaulting to light
 */
function getSavedTheme(): "light" | "dark" {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved === "dark" ? "dark" : "light"; // Default to light
  } catch (error) {
    console.warn("Failed to read theme preference:", error);
    return "light";
  }
}

export const ThemeManager = {
  /**
   * Initialize theme on page load - call this as early as possible
   * to avoid flash of incorrect theme
   */
  init(): void {
    const savedTheme = getSavedTheme();
    applyTheme(savedTheme);
  },

  /**
   * Toggle between light and dark themes
   */
  toggle(): void {
    const currentTheme = this.getCurrentTheme();
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    this.setTheme(newTheme);
  },

  /**
   * Set specific theme
   */
  setTheme(theme: "light" | "dark"): void {
    applyTheme(theme);
    saveTheme(theme);
  },

  /**
   * Get current active theme
   */
  getCurrentTheme(): "light" | "dark" {
    return document.documentElement.classList.contains(DARK_CLASS) ? "dark" : "light";
  },

  /**
   * Check if dark mode is currently active
   */
  isDarkMode(): boolean {
    return this.getCurrentTheme() === "dark";
  },
};

// Initialize theme immediately when module loads
ThemeManager.init();
