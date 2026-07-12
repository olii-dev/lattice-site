const { modalFetch } = require("./_modal");

module.exports = {
  maxDuration: 60,
};

module.exports.default = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const data = await modalFetch("/warm");
    return res.status(200).json(data);
  } catch (e) {
    const status = e.status || 502;
    return res.status(status).json({
      error: e.message || "Warm failed",
    });
  }
};
