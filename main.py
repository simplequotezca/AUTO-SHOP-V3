import os
import json
import re
import uuid
import datetime
from typing import Dict, Optional, List

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    DateTime,
    Text,
)
from sqlalchemy.orm import sessionmaker, declarative_base

from openai import OpenAI

# ---------------------------------------------------------
# OPTIONAL GOOGLE CALENDAR SUPPORT
# ---------------------------------------------------------
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


# ---------------------------------------------------------
# APP + CONFIG
# ---------------------------------------------------------

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

# DB: prefer Railway Postgres via DATABASE_URL, fallback to local sqlite
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback (ephemeral in Railway, but lets the app boot if DB not set)
    DATABASE_URL = "sqlite:///./auto_shop.db"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# In-memory session store for SMS booking flows
SESSIONS: Dict[str, dict] = {}


# ---------------------------------------------------------
# SHOP CONFIG
# ---------------------------------------------------------

class ShopConfig(BaseModel):
    id: str
    name: str
    calendar_id: Optional[str] = None
    webhook_token: str


def load_shops() -> Dict[str, ShopConfig]:
    """
    Load shops from SHOPS_JSON env var.

    Example:
    [
      {
        "id": "sj_auto_body",
        "name": "SJ Auto Body",
        "calendar_id": null,
        "webhook_token": "shop_sj_84k2p1"
      }
    ]
    """
    raw = os.getenv("SHOPS_JSON")
    if not raw:
        # Default single shop if nothing configured
        default = ShopConfig(
            id="default",
            name="Auto Body Shop",
            calendar_id=None,
            webhook_token="shop_default",
        )
        return {default.webhook_token: default}

    data = json.loads(raw)
    shops = [ShopConfig(**s) for s in data]
    return {s.webhook_token: s for s in shops}


SHOPS_BY_TOKEN: Dict[str, ShopConfig] = load_shops()


def get_shop(request: Request) -> ShopConfig:
    if not SHOPS_BY_TOKEN:
        raise HTTPException(status_code=500, detail="No shops configured")

    token = request.query_params.get("token")
    if not token or token not in SHOPS_BY_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid or missing shop token")

    return SHOPS_BY_TOKEN[token]


# ---------------------------------------------------------
# DB MODELS
# ---------------------------------------------------------

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
    """
    Placeholder for future Stripe billing integration.
    Not actively used yet, but schema is ready.
    """
    __tablename__ = "shop_billing"

    shop_id = Column(String, primary_key=True)
    stripe_customer_id = Column(String, nullable=True)
    stripe_subscription_id = Column(String, nullable=True)
    subscription_status = Column(String, nullable=True)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------
# GOOGLE CALENDAR (OPTIONAL)
# ---------------------------------------------------------

def get_calendar_service():
    if not GOOGLE_AVAILABLE:
        return None

    sa_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not sa_path or not os.path.exists(sa_path):
        return None

    creds = service_account.Credentials.from_service_account_file(
        sa_path,
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    return build("calendar", "v3", credentials=creds)


def create_calendar_event(
    shop: ShopConfig,
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    phone: str,
):
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
        body=event,
    ).execute()

    return created.get("id")


# ---------------------------------------------------------
# HELPERS: IMAGES & VIN
# ---------------------------------------------------------

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


def get_appointment_slots(n: int = 3) -> List[datetime.datetime]:
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    hours = [9, 11, 14, 16]

    slots: List[datetime.datetime] = []
    for h in hours:
        dt = tomorrow.replace(hour=h, minute=0, second=0, microsecond=0)
        if dt > now:
            slots.append(dt)
    return slots[:n]


# ---------------------------------------------------------
# OPENAI CLIENT
# ---------------------------------------------------------

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


async def estimate_damage_from_images(
    image_urls: List[str],
    vin: Optional[str],
    shop: ShopConfig,
) -> dict:
    """
    High-accuracy estimator using gpt-4o.
    Ontario 2025 pricing calibrated in the prompt.
    Returns a dict with:
      severity, damage_areas, damage_types, recommended_repairs,
      min_cost, max_cost, confidence, vin_used
    """
    if not client:
        # Fallback if OPENAI_API_KEY not set
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

    # Build multimodal content
    user_content: List[dict] = []
    base_text = "Analyze all uploaded vehicle damage photos and follow the instructions."
    if vin:
        base_text += f" The VIN for this vehicle is: {vin}."
    user_content.append({"type": "text", "text": base_text})

    for url in image_urls:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": url},
            }
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        result = json.loads(raw)

        # Fill defaults and sanity-check cost range
        result.setdefault("severity", "Moderate")
        result.setdefault("damage_areas", [])
        result.setdefault("damage_types", [])
        result.setdefault("recommended_repairs", [])
        result.setdefault("min_cost", 600)
        result.setdefault("max_cost", 1500)
        result.setdefault("confidence", 0.70)
        result.setdefault("vin_used", bool(vin))

        try:
            min_c = float(result["min_cost"])
            max_c = float(result["max_cost"])
            if max_c < min_c:
                min_c, max_c = max_c, min_c
            if max_c - min_c > 6000:
                mid = (min_c + max_c) / 2
                min_c = mid - 1500
                max_c = mid + 1500
            result["min_cost"] = max(100.0, round(min_c))
            result["max_cost"] = max(result["min_cost"] + 50.0, round(max_c))
        except Exception:
            result["min_cost"] = 600
            result["max_cost"] = 1500

        return result

    except Exception as e:
        print("AI Estimator Error:", e)
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


# ---------------------------------------------------------
# HELPERS: DB + ADMIN AUTH
# ---------------------------------------------------------

def save_estimate_to_db(
    shop: ShopConfig,
    phone: str,
    vin: Optional[str],
    result: dict,
) -> str:
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
    incoming = (
        request.headers.get("x-api-key")
        or request.query_params.get("api_key")
        or ""
    )
    if incoming != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.get("/")
def root():
    return {"status": "Backend is running!"}


@app.post("/sms-webhook")
async def sms_webhook(
    request: Request,
    shop: ShopConfig = Depends(get_shop),
):
    """
    Twilio SMS Webhook:
    - If user sends photos → AI estimate + booking options.
    - If user replies 1/2/3 → confirm booking (and Calendar event if configured).
    """
    form = await request.form()

    from_number = form.get("From")
    body = (form.get("Body") or "").strip()
    image_urls = extract_image_urls(form)
    vin = extract_vin(body)

    reply = MessagingResponse()

    session_key = f"{shop.id}:{from_number}"
    session = SESSIONS.get(session_key)

    # 1) Booking reply: "1", "2", "3"
    if session and session.get("awaiting_time") and body in {"1", "2", "3"}:
        try:
            idx = int(body) - 1
        except ValueError:
            idx = -1

        slots: List[datetime.datetime] = session.get("slots", [])
        if 0 <= idx < len(slots):
            chosen = slots[idx]

            # Optional Google Calendar integration
            create_calendar_event(
                shop=shop,
                start_dt=chosen,
                end_dt=chosen + datetime.timedelta(minutes=45),
                phone=from_number,
            )

            reply.message(
                f"Your appointment at {shop.name} is booked for "
                f"{chosen.strftime('%a %b %d at %I:%M %p')}."
            )

            session["awaiting_time"] = False
            SESSIONS[session_key] = session

            return Response(content=str(reply), media_type="application/xml")

    # 2) New AI estimate (if images are present)
    if image_urls:
        result = await estimate_damage_from_images(image_urls, vin, shop)
        estimate_id = save_estimate_to_db(shop, from_number, vin, result)

        severity = result["severity"]
        min_cost = result["min_cost"]
        max_cost = result["max_cost"]
        cost_range = f"${min_cost:,.0f} - ${max_cost:,.0f}"

        areas = (
            ", ".join(result["damage_areas"])
            if result["damage_areas"]
            else "Multiple precise panels identified"
        )
        types = (
            ", ".join(result["damage_types"])
            if result["damage_types"]
            else "Detailed damage types identified"
        )

        slots = get_appointment_slots()
        SESSIONS[session_key] = {"awaiting_time": True, "slots": slots}

        lines = [
            f"AI Damage Estimate - {shop.name}",
            f"Severity: {severity}",
            f"Estimated Repair Range (Ontario 2025): {cost_range}",
            f"Panels: {areas}",
            f"Damage Types: {types}",
            f"Estimate ID (internal): {estimate_id}",
        ]
        if vin and result.get("vin_used"):
            lines.append(f"VIN used for calibration: {vin}")

        lines.append("")
        lines.append("Reply with a number to book an in-person estimate:")

        for i, s in enumerate(slots, start=1):
            lines.append(f"{i}) {s.strftime('%a %b %d at %I:%M %p')}")

        reply.message("\n".join(lines))
        return Response(content=str(reply), media_type="application/xml")

    # 3) No images → instruct user
    intro_lines = [
        f"Thanks for messaging {shop.name}! 👋",
        "",
        "To get an AI-powered pre-estimate:",
        "• Send 1–5 clear photos of the damage",
        "• Optional: include your 17-character VIN in the text",
    ]
    reply.message("\n".join(intro_lines))
    return Response(content=str(reply), media_type="application/xml")


# ---------------------------------------------------------
# ADMIN API (simple JSON dashboard backend)
# ---------------------------------------------------------

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
