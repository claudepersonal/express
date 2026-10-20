# Launch status — Vite + FastAPI

The Vite/FastAPI migration is committed in PR #18. The production Docker build now
preserves the current backend’s pinned, SHA-256 verified YOLO detector provisioning.

Railway project: cardcrop-ai-suite. Service: cardcrop-v2. Existing domain:
https://cardcrop-v2-production.up.railway.app

Launch was attempted. Railway returned: “Your trial has expired. Please select a plan
to continue using Railway.” Service configuration/source can be saved, but that response
did not start a build. Deployment read-back still shows the old sleeping September 24
release. This is not a launched Vite/FastAPI app. The existing /data volume is preserved.

To unblock deployment, the account owner must activate a Railway plan. No billing
change was made. The source is prepared for the existing service; a new stack or
placeholder deployment is not counted as completion.

Local verification: nine focused Node tests and syntax checks pass. The Vite build,
FastAPI/OpenCV integration tests and browser smoke remain unexecuted here because
runtime dependencies and Chromium are unavailable and downloads are restricted.

Other capabilities: HF connection supplies read/jobs scopes, not Space write scope;
Space discovery/docs operations are unavailable. AppDeploy’s returned backend contract
is TypeScript rather than Python FastAPI. Neither establishes a compatible replacement
for this persistent Python deployment.
