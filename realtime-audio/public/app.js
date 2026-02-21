function mustGetElement(id) {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Missing required element: ${id}`);
  }
  return el;
}

const connectBtn = mustGetElement("connectBtn");
const pttBtn = mustGetElement("pttBtn");
const hangupBtn = mustGetElement("hangupBtn");
const statusEl = mustGetElement("status");
const logEl = mustGetElement("log");
const remoteAudioEl = mustGetElement("remoteAudio");

let pc = null;
let dc = null;
let localStream = null;
let micTrack = null;
let pttActive = false;
let connected = false;
let responseInProgress = false;
let pendingResponseAfterCancel = false;
let pttStartedAtMs = 0;

const MIN_AUDIO_MS = 120;

function setStatus(text) {
  statusEl.textContent = text;
}

function log(kind, text) {
  const p = document.createElement("p");
  p.className = kind;
  p.textContent = text;
  logEl.appendChild(p);
  logEl.scrollTop = logEl.scrollHeight;
}

function sendEvent(event) {
  if (!dc || dc.readyState !== "open") {
    return;
  }
  dc.send(JSON.stringify(event));
}

function setUiConnected(isConnected) {
  connected = isConnected;
  connectBtn.disabled = isConnected;
  connectBtn.classList.toggle("connected", isConnected);
  pttBtn.disabled = !isConnected;
  hangupBtn.disabled = !isConnected;
}

function stopPttIfNeeded() {
  if (pttActive) {
    endPushToTalk();
  }
}

async function connect() {
  setStatus("Requesting microphone and session token...");

  localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  micTrack = localStream.getAudioTracks()[0];
  micTrack.enabled = false;

  const tokenRes = await fetch("/token");
  if (!tokenRes.ok) {
    throw new Error(`Token request failed (${tokenRes.status})`);
  }

  const tokenData = await tokenRes.json();
  const ephemeralKey = tokenData?.value || tokenData?.client_secret?.value;
  if (!ephemeralKey) {
    throw new Error("No ephemeral key found in /token response");
  }

  pc = new RTCPeerConnection();
  pc.ontrack = (event) => {
    remoteAudioEl.srcObject = event.streams[0];
  };
  pc.onconnectionstatechange = () => {
    if (pc && (pc.connectionState === "failed" || pc.connectionState === "closed")) {
      disconnect();
    }
  };

  dc = pc.createDataChannel("oai-events");
  dc.onopen = () => {
    log("meta", "Data channel open.");
    setStatus("Connected. Hold button or Space to talk.");
    sendEvent({
      type: "session.update",
      session: {
        type: "realtime",
        output_modalities: ["audio"],
        instructions:
          "You are a concise voice assistant for a push-to-talk demo. Keep answers brief unless asked for detail."
      }
    });
  };

  dc.onmessage = (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      log("meta", "Received non-JSON realtime event.");
      return;
    }

    if (msg.type === "response.created") {
      responseInProgress = true;
      setStatus("Assistant speaking...");
    }

    if (msg.type === "response.done") {
      responseInProgress = false;
      if (pendingResponseAfterCancel && connected && !pttActive) {
        pendingResponseAfterCancel = false;
        createAssistantResponse();
        return;
      }
      setStatus("Connected. Hold button or Space to talk.");
    }

    if (msg.type === "conversation.item.input_audio_transcription.completed") {
      log("user", `You: ${msg.transcript}`);
    }

    if (msg.type === "response.audio_transcript.done" && msg.transcript) {
      log("assistant", `Assistant: ${msg.transcript}`);
    }

    if (msg.type === "error") {
      const errorMessage = msg.error?.message || "unknown error";
      if (errorMessage.includes("active response in progress")) {
        responseInProgress = true;
      } else {
        responseInProgress = false;
        pendingResponseAfterCancel = false;
      }
      log("meta", `Error: ${errorMessage}`);
    }
  };

  pc.addTrack(micTrack, localStream);

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const sdpRes = await fetch("https://api.openai.com/v1/realtime/calls", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${ephemeralKey}`,
      "Content-Type": "application/sdp"
    },
    body: offer.sdp
  });

  if (!sdpRes.ok) {
    throw new Error(`SDP exchange failed (${sdpRes.status})`);
  }

  const answerSdp = await sdpRes.text();
  await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });

  setUiConnected(true);
  setStatus("Connected. Hold button or Space to talk.");
}

function beginPushToTalk() {
  if (!connected || !micTrack || pttActive) {
    return;
  }

  if (responseInProgress) {
    sendEvent({ type: "response.cancel" });
    sendEvent({ type: "output_audio_buffer.clear" });
    log("meta", "Interrupted assistant output.");
  }

  pttActive = true;
  pttStartedAtMs = performance.now();
  pttBtn.classList.add("active");
  micTrack.enabled = true;
  setStatus("Listening... release to send turn.");
}

function createAssistantResponse() {
  if (!connected || !dc || dc.readyState !== "open" || responseInProgress) {
    return;
  }

  responseInProgress = true;
  sendEvent({
    type: "response.create",
    response: {
      output_modalities: ["audio"]
    }
  });

  setStatus("Waiting for assistant...");
}

function endPushToTalk() {
  if (!connected || !micTrack || !pttActive) {
    return;
  }

  pttActive = false;
  pttBtn.classList.remove("active");
  micTrack.enabled = false;

  const durationMs = performance.now() - pttStartedAtMs;
  if (durationMs < MIN_AUDIO_MS) {
    setStatus("Press and hold a bit longer, then release to send.");
    return;
  }

  if (responseInProgress) {
    pendingResponseAfterCancel = true;
    setStatus("Waiting for interruption to finish...");
    return;
  }

  createAssistantResponse();
}

function disconnect() {
  stopPttIfNeeded();

  if (dc) {
    dc.close();
    dc = null;
  }

  if (pc) {
    pc.close();
    pc = null;
  }

  if (localStream) {
    localStream.getTracks().forEach((t) => t.stop());
    localStream = null;
    micTrack = null;
  }

  responseInProgress = false;
  pendingResponseAfterCancel = false;
  pttStartedAtMs = 0;
  remoteAudioEl.srcObject = null;
  setUiConnected(false);
  setStatus("Disconnected");
}

connectBtn.addEventListener("click", async () => {
  connectBtn.disabled = true;
  try {
    await connect();
    log("meta", "Realtime session established.");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus("Connection failed");
    log("meta", `Connection error: ${message}`);
    setUiConnected(false);
    disconnect();
    connectBtn.disabled = false;
  }
});

hangupBtn.addEventListener("click", () => {
  disconnect();
});

pttBtn.addEventListener("mousedown", beginPushToTalk);
pttBtn.addEventListener("mouseup", endPushToTalk);
pttBtn.addEventListener("mouseleave", endPushToTalk);
pttBtn.addEventListener("touchstart", (e) => {
  e.preventDefault();
  beginPushToTalk();
});
pttBtn.addEventListener("touchend", (e) => {
  e.preventDefault();
  endPushToTalk();
});
pttBtn.addEventListener("touchcancel", (e) => {
  e.preventDefault();
  endPushToTalk();
});

window.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    beginPushToTalk();
  }
});

window.addEventListener("keyup", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    endPushToTalk();
  }
});

window.addEventListener("beforeunload", () => {
  disconnect();
});
