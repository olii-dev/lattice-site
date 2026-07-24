const { modalFetch } = require("./_modal");

module.exports = {
  maxDuration: 60,
};

module.exports.default = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const model = (req.body && req.body.model) || "all";

  try {
    const data = await modalFetch("/warm", { model }, model === "pulse2" ? "pulse2" : undefined);
    return res.status(200).json(data);
  } catch (e) {
    const status = e.status || 502;
    return res.status(status).json({
      error: e.message || "Warm failed",
    });
  }
};
