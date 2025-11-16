# Auto-Shop AI Backend (Railway + Twilio)

FastAPI backend for AI damage estimates using **gpt-4o** and PostgreSQL.

## Files

- main.py        -> FastAPI app + Twilio webhook + OpenAI estimator
- requirements.txt
- Procfile

## Environment Variables (on Railway)

Set these on your backend service:

- OPENAI_API_KEY   = your OpenAI API key (`sk-...`)
- ADMIN_API_KEY    = any random strong string (for /admin endpoints)
- DATABASE_URL     = copied from Railway Postgres service
- SHOPS_JSON       = JSON array of shops, e.g.:

```json
[
  {
    "id": "sj_auto_body",
    "name": "SJ Auto Body",
    "calendar_id": null,
    "webhook_token": "shop_sj_84k2p1"
  }
]
