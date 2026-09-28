'use strict';
const { createApp } = require('./app');

const PORT = Number(process.env.PORT || 8080);
const app = createApp();

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Card Restoration Lab listening on :${PORT}`);
  app.checkModel().then(m => console.log('restore model:', JSON.stringify(m)));
});
