# Dashboard video recorder

With the dashboard running on `http://127.0.0.1:8710`:

```bash
cd combined/demo
npm install
npx playwright install chromium ffmpeg
npm run record -- /absolute/path/icu_digital_twin_demo.webm
```

The recorder selects patient 21 at ICU hour 24 and captures the transition through forecast, active alert, watch, and stable states. Set `DASHBOARD_URL` to record another deployment and `CHROMIUM_PATH` to use an existing Chrome or Chromium executable.

Convert the WebM recording to a broadly compatible MP4 with an FFmpeg build containing `libx264`:

```bash
ffmpeg -i icu_digital_twin_demo.webm -c:v libx264 -crf 22 \
  -pix_fmt yuv420p -movflags +faststart -an icu_digital_twin_demo.mp4
```
