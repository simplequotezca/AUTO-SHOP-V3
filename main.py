from fastapi import FastAPI, Request, Depends, HTTPException, Header
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
import datetime
import os
import json
import httpx
from typing import Dict, Optional, List
from google.oauth2 import service_account
from googleapiclient.discovery import build
import re
import uuid

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    DateTime,
    Text,
)
from sqlalchemy.orm import sessionmaker, declarative_base
import stripe

# Optional PDF support (won't crash if not installed)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ============================================================
# DATABASE & STRIPE CONFIG (POSTGRESQL via DATABASE_URL)
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. On Railway, add it in the Variables tab, "
        "pointing to your Postgres instance connection string."
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")  # subscription price id
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")  # for dashboard API auth
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# ============================================================
# SHOP CONFIG
# ============================================================


class ShopConfig(BaseModel):
    id: str
    name: str
    calendar_id: Optional[str] = None
    webhook_token: str


def load_shops() -> Dict[str, ShopConfig]:
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        return {}
    data = json.loads(raw)
    return {s["webhook_token"]: ShopConfig(**s) for s in data}


SHOPS_BY_TOKEN: Dict[str, ShopConfig] = load_shops()
SESSIONS: Dict[str, dict] = {}


def get_shop(request: Request) -> ShopConfig:
    """
    Resolve shop from the `?token=...` query parameter.
    """
    if not SHOPS_BY_TOKEN:
        # Fallback single-shop config so local dev still works
        return ShopConfig(
            id="default",
            name="Auto Body Shop",
            calendar_id=None,
            webhook_token=""
        )
    token = request.query_params.get("token")
    if not token or token not in SHOPS_BY_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing shop token")
    return SHOPS_BY_TOKEN[token]


# ============================================================
# DATABASE MODELS
# ============================================================


class Estimate(Base):
    __tablename__ = "estimates"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    shop_id = Column(String, index=True)
    customer_phone = Column(String, index=True)
    severity = Column(String)
    damage_areas = Column(Text)          # comma-separated
    damage_types = Column(Text)          # comma-separated
    recommended_repairs = Column(Text)   # comma-separated
    min_cost = Column(Float)
    max_cost = Column(Float)
    confidence = Column(Float)
    vin = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ShopBilling(Base):
    __tablename__ = "shop_billing"

    shop_id = Column(String, primary_key=True)
    stripe_customer_id = Column(String, nullable=True)
    stripe_subscription_id = Column(String, nullable=True)
    subscription_status = Column(String, nullable=True)


@app.on_event("startup")
def on_startup() -> None:
    # Create tables if they do not exist
    Base.metadata.create_all(bind=engine)


# ============================================================
# GOOGLE CALENDAR (OPTIONAL)
# ============================================================


def get_calendar_service():
    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_path or not os.path.exists(sa_path):
        return None
    creds = service_account.Credentials.from_service_account_file(
        sa_path,
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    return build("calendar", "v3", credentials=creds)


def create_calendar_event(shop: ShopConfig, start_dt, end_dt, phone: str):
    service = get_calendar_service()
    if not service or not shop.calendar_id:
        return None

    event = {
        "summary": f"Estimate appointment - {shop.name}",
        "description": f"Customer phone: {phone}",
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "America/Toronto"},
        "end": {"dateTime": end_dt.isoformat(), "timeZone": "America/Toronto"},
    }

    created = service.events().insert(
        calendarId=shop.calendar_id,
        body=event
    ).execute()

    return created.get("id")


# ============================================================
# HELPERS: IMAGES + VIN
# ============================================================

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")


def extract_image_urls(form) -> List[str]:
    urls: List[str] = []
    i = 0
    while True:
        key = f"MediaUrl{i}"
        url = form.get(key)
        if not url:
            break
        urls.append(url)
        i += 1
    return urls


def extract_vin(text: str) -> Optional[str]:
    if not text:
        return None
    match = VIN_PATTERN.search(text.upper())
    if match:
        return match.group(1)
    return None


# ============================================================
# AI DAMAGE ESTIMATION (MULTI-IMAGE, ONTARIO 2025)
# ============================================================


async def estimate_damage_from_images(image_urls: List[str], vin: Optional[str], shop: ShopConfig):
    api_key = OPENAI_API_KEY
    if not api_key:
        # Basic fallback so the system still works in demo mode
        return {
            "severity": "Moderate",
            "damage_areas": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600,
            "max_cost": 1500,
            "confidence": 0.60,
            "vin_used": False,
        }

    prompt = """
You are a certified Ontario (Canada) auto-body damage estimator in the year 2025
with 15+ years of experience. You estimate collision and cosmetic repairs
for retail customers (not insurance DRP discounts).

You are given multiple photos of vehicle damage, and possibly a VIN.

Follow this reasoning process INTERNALLY, then output ONLY JSON.

STEP 1: Identify damaged panels
Choose from:
- front bumper upper
- front bumper lower
- rear bumper upper
- rear bumper lower
- left fender
- right fender
- left front door
- right front door
- left rear door
- right rear door
- hood
- trunk
- left quarter panel
- right quarter panel
- rocker panel
- grille area
- headlight area
- taillight area

Be specific. Never say "general damage".

STEP 2: Identify damage types
Choose all that apply:
- dent
- crease dent
- sharp dent
- paint scratch
- deep scratch
- paint scuff
- paint transfer
- crack
- plastic tear
- bumper deformation
- metal distortion
- misalignment
- rust exposure

STEP 3: Suggest repair methods
Choose from:
- PDR (paintless dent repair)
- panel repair + paint
- bumper repair + paint
- bumper replacement
- panel replacement
- blend adjacent panels
- recalibration (sensors/cameras)
- refinish only (no structural repair)

STEP 4: Ontario 2025 pricing calibration (CAD)
Use typical Ontario retail pricing:

- PDR: 150–600
- Panel repaint: 350–900
- Panel repair + repaint: 600–1600
- Bumper repaint: 400–900
- Bumper repair + repaint: 750–1400
- Bumper replacement: 800–2000
- Door replacement: 800–2200
- Quarter panel repair: 900–2500
- Quarter panel replacement: 1800–4800
- Hood repaint: 400–900
- Hood replacement: 600–2200

Rules:
- Minor damage → low end
- Moderate → mid range
- Severe or multiple panels → high end or sum across panels
- If multiple panels clearly damaged, sum realistic operations
- If VIN suggests luxury/EV/aluminum, bias 15–30% higher

STEP 5: VIN usage
If a VIN is provided:
- Infer rough segment (economy / mid-range / luxury / truck / EV)
- Adjust cost band appropriately

STEP 6: Output JSON ONLY
Return strictly this JSON (no extra text):

{
  "severity": "Minor" | "Moderate" | "Severe",
  "damage_areas": [ "front bumper lower", "right fender", ... ],
  "damage_types": [ "dent", "paint scuff", ... ],
  "recommended_repairs": [ "bumper repair + paint", "panel repair + paint", ... ],
  "min_cost": number,
  "max_cost": number,
  "confidence": number,
  "vin_used": boolean
}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    content: List[dict] = []
    main_text = "Analyze all uploaded vehicle damage photos and follow the instructions."
    if vin:
        main_text += f" The VIN for this vehicle is: {vin}."
    content.append({"type": "text", "text": main_text})

    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        resp.raise_for_status()
        data = resp.json()
        output = json.loads(data["choices"][0]["message"]["content"])

        output.setdefault("severity", "Moderate")
        output.setdefault("damage_areas", [])
        output.setdefault("damage_types", [])
        output.setdefault("recommended_repairs", [])
        output.setdefault("min_cost", 600)
        output.setdefault("max_cost", 1500)
        output.setdefault("confidence", 0.70)
        output.setdefault("vin_used", bool(vin))

        # Sanity clamp on cost range
        try:
            min_c = float(output["min_cost"])
            max_c = float(output["max_cost"])
            if max_c < min_c:
                min_c, max_c = max_c, min_c
            if max_c - min_c > 6000:
                mid = (min_c + max_c) / 2
                min_c = mid - 1500
                max_c = mid + 1500
            output["min_cost"] = max(100.0, round(min_c))
            output["max_cost"] = max(output["min_cost"] + 50.0, round(max_c))
        except Exception:
            output["min_cost"] = 600
            output["max_cost"] = 1500

        return output

    except Exception as e:
        # Network / API error fallback
        print("AI Estimator Error (Ontario calibrated):", e)
        return {
            "severity": "Moderate",
            "damage_areas": [],
            "damage_types": [],
            "recommended_repairs": [],
            "min_cost": 600,
            "max_cost": 1500,
            "confidence": 0.55,
            "vin_used": bool(vin),
        }


# ============================================================
# PDF ESTIMATE GENERATOR (OPTIONAL)
# ============================================================


def generate_estimate_pdf(shop: ShopConfig, phone: str, result: dict) -> Optional[str]:
    if not REPORTLAB_AVAILABLE:
        return None

    safe_phone = phone.replace("+", "").replace(" ", "")
    file_name = f"/tmp/estimate_{shop.id}_{safe_phone}.pdf"

    c = canvas.Canvas(file_name, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, f"{shop.name} - AI Damage Estimate")

    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Customer phone: {phone}")
    y -= 20

    c.drawString(50, y, f"Severity: {result.get('severity', 'N/A')}")
    y -= 20
    c.drawString(
        50,
        y,
        f"Estimated Cost: ${result.get('min_cost', 0):,.0f} - ${result.get('max_cost', 0):,.0f}",
    )
    y -= 30

    areas = ", ".join(result.get("damage_areas", [])) or "N/A"
    types = ", ".join(result.get("damage_types", [])) or "N/A"
    repairs = ", ".join(result.get("recommended_repairs", [])) or "N/A"

    c.drawString(50, y, f"Damage Areas: {areas}")
    y -= 20
    c.drawString(50, y, f"Damage Types: {types}")
    y -= 20
    c.drawString(50, y, f"Recommended Repairs: {repairs}")
    y -= 20

    conf = result.get("confidence", 0.0)
    c.drawString(50, y, f"Model confidence: {conf:.2f}")
    y -= 30

    c.drawString(
        50,
        y,
        "Note: This is an AI-assisted pre-estimate. Final pricing may vary after in-person inspection.",
    )
    c.showPage()
    c.save()
    return file_name


# ============================================================
# HELPERS: SAVE ESTIMATE + ADMIN AUTH
# ============================================================


def save_estimate_to_db(shop: ShopConfig, phone: str, vin: Optional[str], result: dict) -> str:
    db = SessionLocal()
    try:
        est = Estimate(
            shop_id=shop.id,
            customer_phone=phone,
            severity=result.get("severity"),
            damage_areas=", ".join(result.get("damage_areas", [])),
            damage_types=", ".join(result.get("damage_types", [])),
            recommended_repairs=", ".join(result.get("recommended_repairs", [])),
            min_cost=result.get("min_cost"),
            max_cost=result.get("max_cost"),
            confidence=result.get("confidence"),
            vin=vin,
        )
        db.add(est)
        db.commit()
        db.refresh(est)
        return est.id
    finally:
        db.close()


def require_admin(request: Request):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not configured")
    incoming = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if incoming != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


# ============================================================
# APPOINTMENT SLOTS
# ============================================================


def get_appointment_slots(n: int = 3):
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    hours = [9, 11, 14, 16]

    slots = []
    for h in hours:
        dt = tomorrow.replace(hour=h, minute=0, second=0, microsecond=0)
        if dt > now:
            slots.append(dt)
    return slots[:n]


# ============================================================
# ROUTES: HEALTHCHECK
# ============================================================


@app.get("/")
def root():
    return {"status": "Backend is running!"}


# ============================================================
# ROUTE: TWILIO SMS WEBHOOK
# ============================================================


@app.post("/sms-webhook")
async def sms_webhook(request: Request, shop: ShopConfig = Depends(get_shop)):
    form = await request.form()
    from_number = form.get("From")
    body = (form.get("Body") or "").strip()

    image_urls = extract_image_urls(form)
    vin = extract_vin(body)

    reply = MessagingResponse()

    session_key = f"{shop.id}:{from_number}"
    session = SESSIONS.get(session_key)

    # Booking selection flow
    if session and session.get("awaiting_time") and body in {"1", "2", "3"}:
        idx = int(body) - 1
        slots = session["slots"]

        if 0 <= idx < len(slots):
            chosen = slots[idx]

            create_calendar_event(
                shop=shop,
                start_dt=chosen,
                end_dt=chosen + datetime.timedelta(minutes=45),
                phone=from_number,
            )

            reply.message(
                f"Your appointment is booked for {chosen.strftime('%a %b %d at %I:%M %p')}."
            )

            session["awaiting_time"] = False
            SESSIONS[session_key] = session

            return Response(content=str(reply), media_type="application/xml")

    # Multi-image AI estimate
    if image_urls:
        result = await estimate_damage_from_images(image_urls, vin, shop)

        estimate_id = save_estimate_to_db(shop, from_number, vin, result)

        severity = result["severity"]
        min_cost = result["min_cost"]
        max_cost = result["max_cost"]
        cost_range = f"${min_cost:,.0f} - ${max_cost:,.0f}"

        areas = ", ".join(result["damage_areas"]) if result["damage_areas"] else "specific panels detected"
        types = ", ".join(result["damage_types"]) if result["damage_types"] else "detailed damage types detected"

        slots = get_appointment_slots()
        SESSIONS[session_key] = {"awaiting_time": True, "slots": slots}

        lines = [
            f"AI Damage Estimate for {shop.name}",
            f"Severity: {severity}",
            f"Estimated Cost (Ontario 2025): {cost_range}",
            f"Panels: {areas}",
            f"Damage Types: {types}",
            f"Estimate ID (internal): {estimate_id}",
        ]

        if vin and result.get("vin_used"):
            lines.append(f"VIN used for calibration: {vin}")

        lines.append("")
        lines.append("Reply with a number to book an in-person estimate:")

        for i, s in enumerate(slots, 1):
            lines.append(f"{i}) {s.strftime('%a %b %d at %I:%M %p')}")

        # Generate PDF quietly (if available)
        generate_estimate_pdf(shop, from_number, result)

        reply.message("\n".join(lines))
        return Response(content=str(reply), media_type="application/xml")

    # Default prompt (no images)
    intro = [
        f"Thanks for messaging {shop.name}.",
        "",
        "To get an AI-powered pre-estimate:",
        "- Send 1–5 clear photos of the damage",
        "- Optional: include your 17-character VIN in the text",
    ]

    reply.message("\n".join(intro))
    return Response(content=str(reply), media_type="application/xml")


# ============================================================
# ADMIN DASHBOARD API (very lightweight)
# ============================================================


@app.get("/admin/estimates")
def list_estimates(
    request: Request,
    shop_id: Optional[str] = None,
    limit: int = 50,
    skip: int = 0,
):
    require_admin(request)
    db = SessionLocal()
    try:
        q = db.query(Estimate)
        if shop_id:
            q = q.filter(Estimate.shop_id == shop_id)
        q = q.order_by(Estimate.created_at.desc()).offset(skip).limit(limit)
        rows = q.all()
        return [
            {
                "id": e.id,
                "shop_id": e.shop_id,
                "customer_phone": e.customer_phone,
                "severity": e.severity,
                "min_cost": e.min_cost,
                "max_cost": e.max_cost,
                "created_at": e.created_at.isoformat(),
            }
            for e in rows
        ]
    finally:
        db.close()


@app.get("/admin/estimates/{estimate_id}")
def get_estimate(estimate_id: str, request: Request):
    require_admin(request)
    db = SessionLocal()
    try:
        e = db.query(Estimate).filter(Estimate.id == estimate_id).first()
        if not e:
            raise HTTPException(status_code=404, detail="Estimate not found")
        return {
            "id": e.id,
            "shop_id": e.shop_id,
            "customer_phone": e.customer_phone,
            "severity": e.severity,
            "damage_areas": e.damage_areas,
            "damage_types": e.damage_types,
            "recommended_repairs": e.recommended_repairs,
            "min_cost": e.min_cost,
            "max_cost": e.max_cost,
            "confidence": e.confidence,
            "vin": e.vin,
            "created_at": e.created_at.isoformat(),
        }
    finally:
        db.close()


# ============================================================
# STRIPE BILLING ENDPOINTS (minimal / optional)
# ============================================================


class CheckoutRequest(BaseModel):
    shop_id: str
    success_url: str
    cancel_url: str


@app.post("/billing/create-checkout-session")
def create_checkout_session(payload: CheckoutRequest):
    if not STRIPE_API_KEY or not STRIPE_PRICE_ID:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=payload.success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=payload.cancel_url,
            metadata={"shop_id": payload.shop_id},
        )
        return {"checkout_url": session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/billing/stripe-webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret not set")

    body = await request.body()
    try:
        event = stripe.Webhook.construct_event(
            payload=body,
            sig_header=stripe_signature,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {e}")

    event_type = event["type"]
    data = event["data"]["object"]

    db = SessionLocal()
    try:
        if event_type == "checkout.session.completed":
            shop_id = data.get("metadata", {}).get("shop_id")
            subscription_id = data.get("subscription")
            customer_id = data.get("customer")

            if shop_id:
                record = db.query(ShopBilling).filter(ShopBilling.shop_id == shop_id).first()
                if not record:
                    record = ShopBilling(shop_id=shop_id)
                    db.add(record)
                record.stripe_customer_id = customer_id
                record.stripe_subscription_id = subscription_id
                record.subscription_status = "active"
                db.commit()

        elif event_type == "customer.subscription.updated":
            subscription_id = data.get("id")
            status = data.get("status")
            record = db.query(ShopBilling).filter(
                ShopBilling.stripe_subscription_id == subscription_id
            ).first()
            if record:
                record.subscription_status = status
                db.commit()

    finally:
        db.close()

    return {"received": True}


# ============================================================
# LOCAL ENTRYPOINT (Railway uses `python main.py`)
# ============================================================


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

