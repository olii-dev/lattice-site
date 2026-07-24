const { modalFetch } = require("./_modal");

module.exports = {
  maxDuration: 60,
};

const ALLOWED = new Set(["pulse", "pulse2"]);

module.exports.default = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { message, history, model } = req.body || {};
  if (!message || typeof message !== "string" || !message.trim()) {
    return res.status(400).json({ error: "message is required" });
  }

  const modelId = ALLOWED.has(model) ? model : "pulse";

  try {
    const data = await modalFetch("/chat", {
      message: message.trim(),
      history: Array.isArray(history) ? history : [],
      model: modelId,
    }, modelId);
    return res.status(200).json(data);
  } catch (e) {
    const status = e.status || 502;
    return res.status(status).json({
      error: e.message || "Chat failed",
    });
  }
};
