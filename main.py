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

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

app = FastAPI()

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

SHOPS_BY_TOKEN = load_shops()
SESSIONS = {}

def get_shop(request: Request) -> ShopConfig:
    if not SHOPS_BY_TOKEN:
        return ShopConfig(id="default", name="Auto Body Shop", calendar_id=None, webhook_token="")
    token = request.query_params.get("token")
    if not token or token not in SHOPS_BY_TOKEN:
        raise HTTPException(403, "Invalid shop token")
    return SHOPS_BY_TOKEN[token]

class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    shop_id = Column(String)
    customer_phone = Column(String)
    severity = Column(String)
    damage_areas = Column(Text)
    damage_types = Column(Text)
    recommended_repairs = Column(Text)
    min_cost = Column(Float)
    max_cost = Column(Float)
    confidence = Column(Float)
    vin = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ShopBilling(Base):
    __tablename__ = "shop_billing"
    shop_id = Column(String, primary_key=True)
    stripe_customer_id = Column(String)
    stripe_subscription_id = Column(String)
    subscription_status = Column(String)

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

def extract_image_urls(form):
    urls = []
    i = 0
    while True:
        u = form.get(f"MediaUrl{i}")
        if not u:
            break
        urls.append(u)
        i += 1
    return urls

VIN_PATTERN = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
def extract_vin(t):
    if not t:
        return None
    m = VIN_PATTERN.search(t.upper())
    return m.group(1) if m else None

async def estimate_damage_from_images(image_urls, vin, shop):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"severity":"Moderate","damage_areas":[],"damage_types":[],"recommended_repairs":[],"min_cost":600,"max_cost":1500,"confidence":0.6,"vin_used":False}

    prompt = "You are an Ontario 2025 collision estimator. Return JSON only."
    headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}

    content=[{"type":"text","text":"Analyze photos."}]
    if vin:
        content[0]["text"] += f" VIN: {vin}"
    for u in image_urls:
        content.append({"type":"image_url","image_url":{"url":u}})

    payload={
        "model":"gpt-4.1",
        "messages":[{"role":"system","content":prompt},{"role":"user","content":content}],
        "response_format":{"type":"json_object"}
    }
    try:
        async with httpx.AsyncClient(timeout=45) as c:
            resp=await c.post("https://api.openai.com/v1/chat/completions",headers=headers,json=payload)
        out=json.loads(resp.json()["choices"][0]["message"]["content"])
        out.setdefault("min_cost",600)
        out.setdefault("max_cost",1500)
        out["vin_used"]=bool(vin)
        return out
    except:
        return {"severity":"Moderate","damage_areas":[],"damage_types":[],"recommended_repairs":[],"min_cost":600,"max_cost":1500,"confidence":0.55,"vin_used":bool(vin)}

def save_estimate_to_db(shop, phone, vin, result):
    db=SessionLocal()
    try:
        e=Estimate(
            shop_id=shop.id,
            customer_phone=phone,
            severity=result.get("severity"),
            damage_areas=", ".join(result.get("damage_areas",[])),
            damage_types=", ".join(result.get("damage_types",[])),
            recommended_repairs=", ".join(result.get("recommended_repairs",[])),
            min_cost=result.get("min_cost"),
            max_cost=result.get("max_cost"),
            confidence=result.get("confidence"),
            vin=vin
        )
        db.add(e); db.commit(); db.refresh(e)
        return e.id
    finally:
        db.close()

def get_appointment_slots(n=3):
    now=datetime.datetime.now()
    t=now+datetime.timedelta(days=1)
    hours=[9,11,14,16]
    out=[]
    for h in hours:
        dt=t.replace(hour=h,minute=0,second=0,microsecond=0)
        if dt>now:
            out.append(dt)
    return out[:n]

@app.get("/")
def root():
    return {"status":"Backend OK"}

@app.post("/sms-webhook")
async def sms_webhook(request: Request, shop: ShopConfig=Depends(get_shop)):
    form=await request.form()
    from_number=form.get("From")
    body=(form.get("Body") or "").strip()
    imgs=extract_image_urls(form)
    vin=extract_vin(body)
    reply=MessagingResponse()

    key=f"{shop.id}:{from_number}"
    session=SESSIONS.get(key)

    if session and session.get("awaiting_time") and body in {"1","2","3"}:
        i=int(body)-1
        slots=session["slots"]
        if 0<=i<len(slots):
            chosen=slots[i]
            reply.message(f"Appointment booked for {chosen}.")
            session["awaiting_time"]=False
            SESSIONS[key]=session
            return Response(content=str(reply),media_type="application/xml")

    if imgs:
        res=await estimate_damage_from_images(imgs,vin,shop)
        est_id=save_estimate_to_db(shop,from_number,vin,res)
        slots=get_appointment_slots()
        SESSIONS[key]={"awaiting_time":True,"slots":slots}
        lines=[
            f"AI Estimate for {shop.name}",
            f"Severity: {res['severity']}",
            f"Cost: ${res['min_cost']} - ${res['max_cost']}",
            f"Panels: {', '.join(res.get('damage_areas',[]))}",
            f"Damage Types: {', '.join(res.get('damage_types',[]))}",
            f"Estimate ID: {est_id}",
            "",
            "Reply with a number to book:"
        ]
        for idx,s in enumerate(slots,1):
            lines.append(f"{idx}) {s.strftime('%a %b %d %I:%M %p')}")
        reply.message("\n".join(lines))
        return Response(content=str(reply),media_type="application/xml")

    reply.message("Send damage photos for AI estimate.")
    return Response(content=str(reply),media_type="application/xml")