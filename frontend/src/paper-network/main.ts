import "../style.css";
import { ApiService } from "./apiService";
import { PaperNetwork } from "./network";
import {
  retryWithBackoff,
  waitForDOM,
  showInitError,
  isMobileDevice,
  showMobileMessage,
  warmBackend,
} from "../util";
import { addFooter } from "../footer";

/**
 * Initialise the Paper Network application.
 */
async function initialiseApp(): Promise<void> {
  const apiUrl = import.meta.env.VITE_API_URL as string | undefined;
  warmBackend(apiUrl);

  await waitForDOM();

  if (isMobileDevice()) {
    showMobileMessage();
    return;
  }

  if (!apiUrl) throw new Error("Set VITE_API_URL environment variable.");
  const dataService = new ApiService(apiUrl);

  // Initialise with retry logic
  await retryWithBackoff(() => {
    new PaperNetwork(dataService);
    console.log("PaperNetwork initialised successfully");
    return Promise.resolve();
  });

  // Add footer
  addFooter();
}

initialiseApp().catch(showInitError);
