import dotenv from "dotenv";
import express, { type Request, type Response } from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootEnvPath = path.resolve(__dirname, "../../.env");

dotenv.config({ path: rootEnvPath });

const app = express();
const port = Number(process.env.PORT ?? 3000);
const publicDir = path.resolve(__dirname, "../public");

app.use(express.static(publicDir));

app.get("/health", (_req: Request, res: Response) => {
  res.json({ ok: true });
});

app.get("/token", async (_req: Request, res: Response) => {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    res.status(500).json({
      error: "Missing OPENAI_API_KEY in environment"
    });
    return;
  }

  try {
    const response = await fetch("https://api.openai.com/v1/realtime/client_secrets", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        session: {
          type: "realtime",
          model: "gpt-realtime",
          audio: {
            output: {
              voice: "marin"
            }
          },
          instructions:
            "You are a concise assistant in a push-to-talk voice demo. Keep responses short and clear."
        }
      })
    });

    if (!response.ok) {
      const details = await response.text();
      res.status(response.status).json({
        error: "Failed to create realtime session",
        details
      });
      return;
    }

    const data = await response.json();
    res.setHeader("Cache-Control", "no-store");
    res.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({
      error: "Unexpected error creating realtime session",
      details: message
    });
  }
});

app.use((_req: Request, res: Response) => {
  res.sendFile(path.join(publicDir, "index.html"));
});

app.listen(port, () => {
  console.log(`Realtime audio demo listening on http://localhost:${port}`);
});
